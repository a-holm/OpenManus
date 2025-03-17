import asyncio
import time
from collections import deque
from typing import Dict, List, Optional, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)


REASONING_MODELS = ["o1", "o3-mini"]


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # Add token counting related attributes
            self.total_input_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            # Add rate limiting attributes
            self.rpm_limit = getattr(llm_config, "rpm_limit", None)
            self.tpm_limit = getattr(llm_config, "tpm_limit", None)
            self.itpm_limit = getattr(llm_config, "itpm_limit", None)
            self.otpm_limit = getattr(llm_config, "otpm_limit", None)
            self.min_interval = 60 / self.rpm_limit if self.rpm_limit else 0
            self.last_request_time = None
            self.token_tracker = deque()  # (timestamp, total_tokens)
            self.input_token_tracker = deque()  # (timestamp, input_tokens)
            self.output_token_tracker = deque()  # (timestamp, output_tokens)

            # Initialize tokenizer
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # If the model is not in tiktoken's presets, use cl100k_base as default
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the number of tokens in a message list"""
        token_count = 0
        for message in messages:
            # Base token count for each message (according to OpenAI's calculation method)
            token_count += 4  # Base token count for each message

            # Calculate tokens for the role
            if "role" in message:
                token_count += self.count_tokens(message["role"])

            # Calculate tokens for the content
            if "content" in message and message["content"]:
                token_count += self.count_tokens(message["content"])

            # Calculate tokens for tool calls
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    if "function" in tool_call:
                        # Function name
                        if "name" in tool_call["function"]:
                            token_count += self.count_tokens(
                                tool_call["function"]["name"]
                            )
                        # Function arguments
                        if "arguments" in tool_call["function"]:
                            token_count += self.count_tokens(
                                tool_call["function"]["arguments"]
                            )

            # Calculate tokens for tool responses
            if "name" in message and message["name"]:
                token_count += self.count_tokens(message["name"])

            if "tool_call_id" in message and message["tool_call_id"]:
                token_count += self.count_tokens(message["tool_call_id"])

        # Add extra tokens for message format
        token_count += 2  # Extra tokens for message format

        return token_count

    def update_token_count(self, input_tokens: int) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Cumulative Input={self.total_input_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    async def enforce_rate_limits(self) -> None:
        """
        Apply rate limits (RPM, TPM, ITPM, OTPM) before making API calls.
        Waits as needed to comply with configured limits.
        """
        # Skip if no rate limits are configured
        if (
            not self.rpm_limit
            and not self.tpm_limit
            and not self.itpm_limit
            and not self.otpm_limit
        ):
            return

        # Enforce RPM limit
        if self.rpm_limit is not None:
            if self.last_request_time is not None:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
                    logger.info(
                        f"Rate limit pause: waiting {wait_time:.2f}s for RPM limit"
                    )
                    await asyncio.sleep(wait_time)

        current_time = time.time()

        # Enforce TPM limit
        if self.tpm_limit is not None:
            # Remove entries older than 60 seconds
            while self.token_tracker and self.token_tracker[0][0] < current_time - 60:
                self.token_tracker.popleft()

            current_tpm = sum(t for _, t in self.token_tracker)

            # Wait until we're under TPM limit
            while current_tpm >= self.tpm_limit and self.token_tracker:
                oldest_entry_time = self.token_tracker[0][0]
                wait_time = (oldest_entry_time + 60) - current_time
                if wait_time > 0:
                    logger.info(
                        f"Rate limit pause: waiting {wait_time:.2f}s for TPM limit (current TPM: {current_tpm})"
                    )
                    await asyncio.sleep(wait_time)

                current_time = time.time()

                # Prune again after waiting
                while (
                    self.token_tracker and self.token_tracker[0][0] < current_time - 60
                ):
                    self.token_tracker.popleft()

                current_tpm = sum(t for _, t in self.token_tracker)

        # Enforce Input TPM limit
        if self.itpm_limit is not None:
            # Remove entries older than 60 seconds
            while (
                self.input_token_tracker
                and self.input_token_tracker[0][0] < current_time - 60
            ):
                self.input_token_tracker.popleft()

            current_itpm = sum(t for _, t in self.input_token_tracker)

            # Wait until we're under ITPM limit
            while current_itpm >= self.itpm_limit and self.input_token_tracker:
                oldest_entry_time = self.input_token_tracker[0][0]
                wait_time = (oldest_entry_time + 60) - current_time
                if wait_time > 0:
                    logger.info(
                        f"Rate limit pause: waiting {wait_time:.2f}s for ITPM limit (current ITPM: {current_itpm})"
                    )
                    await asyncio.sleep(wait_time)

                current_time = time.time()

                # Prune again after waiting
                while (
                    self.input_token_tracker
                    and self.input_token_tracker[0][0] < current_time - 60
                ):
                    self.input_token_tracker.popleft()

                current_itpm = sum(t for _, t in self.input_token_tracker)

        # Enforce Output TPM limit
        if self.otpm_limit is not None:
            # Remove entries older than 60 seconds
            while (
                self.output_token_tracker
                and self.output_token_tracker[0][0] < current_time - 60
            ):
                self.output_token_tracker.popleft()

            current_otpm = sum(t for _, t in self.output_token_tracker)

            # Wait until we're under OTPM limit
            while current_otpm >= self.otpm_limit and self.output_token_tracker:
                oldest_entry_time = self.output_token_tracker[0][0]
                wait_time = (oldest_entry_time + 60) - current_time
                if wait_time > 0:
                    logger.info(
                        f"Rate limit pause: waiting {wait_time:.2f}s for OTPM limit (current OTPM: {current_otpm})"
                    )
                    await asyncio.sleep(wait_time)

                current_time = time.time()

                # Prune again after waiting
                while (
                    self.output_token_tracker
                    and self.output_token_tracker[0][0] < current_time - 60
                ):
                    self.output_token_tracker.popleft()

                current_otpm = sum(t for _, t in self.output_token_tracker)

    def update_rate_limit_trackers(
        self, tokens_used: int, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """
        Update rate limit tracking after an API call

        Args:
            tokens_used: Total tokens used (input + output)
            input_tokens: Input tokens used (prompt tokens)
            output_tokens: Output tokens used (completion tokens)
        """
        self.last_request_time = time.time()

        # Track token usage if TPM limit is enabled
        if self.tpm_limit is not None:
            self.token_tracker.append((self.last_request_time, tokens_used))

        # Track input token usage if ITPM limit is enabled
        if self.itpm_limit is not None:
            self.input_token_tracker.append((self.last_request_time, input_tokens))

        # Track output token usage if OTPM limit is enabled
        if self.otpm_limit is not None:
            self.output_token_tracker.append((self.last_request_time, output_tokens))

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, Message):
                message = message.to_dict()
            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Apply rate limiting before API call
            await self.enforce_rate_limits()

            if not stream:
                # Non-streaming request
                params["stream"] = False

                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # Update token counts and rate limit trackers
                self.update_token_count(response.usage.prompt_tokens)
                self.update_rate_limit_trackers(
                    response.usage.total_tokens,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

                return response.choices[0].message.content

            # Streaming request, For streaming, update estimated token count before making the request
            self.update_token_count(input_tokens)

            params["stream"] = True
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            estimated_output_tokens = 0
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                # Count tokens in each chunk for better token tracking
                if chunk_message:
                    estimated_output_tokens += self.count_tokens(chunk_message)
                print(chunk_message, end="", flush=True)

            # Update rate limit trackers with more accurate estimate
            total_estimated_tokens = input_tokens + estimated_output_tokens
            self.update_rate_limit_trackers(
                total_estimated_tokens, input_tokens, estimated_output_tokens
            )

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 3000,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Set up the completion request
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Apply rate limiting before API call
            await self.enforce_rate_limits()

            response = await self.client.chat.completions.create(**params)

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print(response)
                raise ValueError("Invalid or empty response from LLM")

            # Update token counts and rate limit trackers
            self.update_token_count(response.usage.prompt_tokens)
            self.update_rate_limit_trackers(
                response.usage.total_tokens,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
                time.sleep(60)
            elif isinstance(oe, APIError):
                logger.error(f"Content: {response}")
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.info(f"Response: {response}")
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise
