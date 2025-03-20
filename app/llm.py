import asyncio
import math
import time
from collections import deque
from typing import Deque, Tuple, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
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
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS: int = 4
    FORMAT_TOKENS: int = 2
    LOW_DETAIL_IMAGE_TOKENS: int = 85
    HIGH_DETAIL_TILE_TOKENS: int = 170

    # Image processing constants
    MAX_SIZE: int = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE: int = 768
    TILE_SIZE: int = 512

    def __init__(self, tokenizer: tiktoken.Encoding) -> None:
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        # Default values when dimensions aren't available or detail level is unknown
        if detail == "high":
            # Default to a 1024x1024 image calculation for high detail
            return self._calculate_high_detail_tokens(1024, 1024)  # 765 tokens
        elif detail == "medium":
            # Default to a medium-sized image for medium detail
            return 1024  # This matches the original default
        else:
            # For unknown detail levels, use medium as default
            return 1024

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, list[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: list[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: list[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    _instances: dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: LLMSettings | None = None
    ) -> "LLM":
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: LLMSettings | None = None
    ) -> None:
        # Initialize only once per instance.
        if hasattr(self, "client"):
            return

        llm_config = llm_config or config.llm
        llm_config = llm_config.get(config_name, llm_config["default"])
        self.model: str = llm_config.model
        self.max_tokens: int = llm_config.max_tokens
        self.temperature: float = llm_config.temperature
        self.api_type: str = llm_config.api_type
        self.api_key: str = llm_config.api_key
        self.api_version: str = llm_config.api_version
        self.base_url: str = llm_config.base_url

        # Token tracking attributes.
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.max_input_tokens: int | None = getattr(
            llm_config, "max_input_tokens", None
        )

        # Rate limiting per-minute limits.
        self.rpm_limit: int | None = getattr(llm_config, "rpm_limit", None)
        self.tpm_limit: int | None = getattr(llm_config, "tpm_limit", None)
        self.itpm_limit: int | None = getattr(llm_config, "itpm_limit", None)
        self.otpm_limit: int | None = getattr(llm_config, "otpm_limit", None)
        self.min_interval: float = 60 / self.rpm_limit if self.rpm_limit else 0
        self.last_request_time: float | None = None

        # Trackers for token-based rate limits.
        self.requests_tracker: Deque[float] = deque()  # Requests last minute.
        self.token_tracker: Deque[Tuple[float, int]] = deque()  # General tracker.
        self.input_token_tracker: Deque[
            Tuple[float, int]
        ] = deque()  # Input tokens per minute.
        self.output_token_tracker: Deque[
            Tuple[float, int]
        ] = deque()  # Output tokens per minute.

        # Lock for updating trackers.
        self._tracker_lock = asyncio.Lock()

        # Initialize the tokenizer.
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        if self.api_type == "azure":
            self.client = AsyncAzureOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        else:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        self.token_counter = TokenCounter(self.tokenizer)

        # Cap max_tokens if it exceeds the output tokens-per-minute limit.
        if self.otpm_limit is not None and self.max_tokens > self.otpm_limit:
            logger.warning(
                f"max_tokens ({self.max_tokens}) exceeds output tokens per minute limit ({self.otpm_limit}). "
                f"Capping max_tokens to {self.otpm_limit}."
            )
            self.max_tokens = self.otpm_limit

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in a given text."""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: list[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    async def update_token_count(
        self, input_tokens: int, output_tokens: int = 0
    ) -> None:
        """
        Update the cumulative token counts and token trackers.

        Args:
            input_tokens (int): The number of input tokens used for the request.
            output_tokens (int, optional): The number of output tokens produced. Defaults to 0.
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        now = time.time()

        async with self._tracker_lock:
            # Update token trackers
            if input_tokens > 0:
                self.input_token_tracker.append((now, input_tokens))
                self.token_tracker.append((now, input_tokens))
            if output_tokens > 0:
                self.output_token_tracker.append((now, output_tokens))
                self.token_tracker.append((now, output_tokens))

            # Clean up expired token records
            while self.token_tracker and now - self.token_tracker[0][0] >= 60:
                self.token_tracker.popleft()
            while (
                self.input_token_tracker and now - self.input_token_tracker[0][0] >= 60
            ):
                self.input_token_tracker.popleft()
            while (
                self.output_token_tracker
                and now - self.output_token_tracker[0][0] >= 60
            ):
                self.output_token_tracker.popleft()

            # Clean up expired request records too for accurate count
            while self.requests_tracker and now - self.requests_tracker[0] >= 60:
                self.requests_tracker.popleft()

            # Just use the length of the tracker for RPM
            current_rpm = len(self.requests_tracker)

        logger.info(
            f"Token usage: Input={input_tokens}, Output={output_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Output={self.total_output_tokens}, "
            f"Total for request={input_tokens + output_tokens}, "
            f"Cumulative Total={self.total_input_tokens + self.total_output_tokens}, "
            f"RPM (requests per minute)={current_rpm}"
        )

    async def wait_for_token_availability(
        self, planned_tokens: int, tracker: Deque[Tuple[float, int]], limit: int
    ) -> None:
        """
        Wait until adding planned_tokens to the given tracker will not exceed the limit.

        This method holds the _tracker_lock for the entire check-and-wait loop. It removes
        any expired token records (older than 60 seconds) and then checks if the sum of the
        tracked tokens plus planned_tokens is within the limit. If it is, it reserves the
        tokens (by appending a new record) and returns.

        Otherwise, it calculates how long to sleep until at least the earliest token record expires,
        logs that wait time, and then awaits the sleep while still holding the lock.
        """
        while True:
            async with self._tracker_lock:
                now = time.time()
                # Remove expired records older than 60 seconds.
                while tracker and now - tracker[0][0] > 60:
                    tracker.popleft()
                current_total = sum(tokens for (_, tokens) in tracker)
                if planned_tokens <= limit - current_total:
                    # Reserve the tokens and return.
                    tracker.append((now, planned_tokens))
                    return
                # Calculate the wait time until the oldest token record expires.
                sleep_time = 60 - (now - tracker[0][0]) if tracker else 0.1
                logger.info(
                    f"Waiting for token availability: planned_tokens={planned_tokens}, "
                    f"current_total={current_total}, limit={limit}. Sleeping for {sleep_time:.2f} seconds"
                )
                await asyncio.sleep(sleep_time)

    def truncate_messages(self, messages: list[dict]) -> list[dict]:
        """
        If the token count of messages greatly exceeds the itpm_limit,
        log a warning and trim older messages until the count fits.

        This naive approach simply drops the earliest messages one by one.
        """
        current_tokens = self.count_message_tokens(messages)
        if self.itpm_limit is not None and current_tokens > self.itpm_limit:
            logger.warning(
                f"Input messages token count ({current_tokens}) exceeds the input tokens per minute "
                f"limit ({self.itpm_limit}). Truncating older messages."
            )
            new_messages = messages.copy()
            while (
                new_messages
                and self.count_message_tokens(new_messages) > self.itpm_limit
            ):
                new_messages.pop(0)
            return new_messages
        return messages

    @staticmethod
    def format_messages(
        messages: list[dict | Message], supports_images: bool = False
    ) -> list[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            list[dict]: List of formatted messages in OpenAI format

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
        formatted_messages: list[dict] = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "tool_calls" in message:
                    formatted_messages.append(message)
                elif "content" in message and message["content"].strip():
                    # Make sure content exist and has non-empty text
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    async def wait_for_request_rate(self) -> None:
        """
        Wait until the number of requests in the past 60 seconds
        is below the allowed RPM limit.
        """
        if not self.rpm_limit:  # Skip if no limit is set
            return

        while True:
            async with self._tracker_lock:
                now = time.time()
                # Remove any expired requests (older than 60 seconds)
                while self.requests_tracker and now - self.requests_tracker[0] >= 60:
                    self.requests_tracker.popleft()

                current_rpm = len(self.requests_tracker)
                logger.info(
                    f"Current RPM before check: {current_rpm}, Limit: {self.rpm_limit}"
                )

                # If adding a new request would not exceed the limit, update the tracker and return
                if current_rpm < self.rpm_limit:
                    self.requests_tracker.append(now)
                    self.last_request_time = now
                    logger.info(
                        f"Request allowed. New RPM: {len(self.requests_tracker)}"
                    )
                    return

                # Otherwise, calculate the time until the oldest request expires
                if self.requests_tracker:
                    sleep_time = (
                        60 - (now - self.requests_tracker[0]) + 0.1
                    )  # Add a small buffer
                else:
                    sleep_time = 0.1  # Should never happen but just in case

                logger.info(
                    f"Enforcing request rate limit: current RPM={current_rpm}, "
                    f"limit={self.rpm_limit}, sleeping {sleep_time:.2f} seconds"
                )

            # Release the lock during sleep to allow other operations
            await asyncio.sleep(sleep_time)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(
        self,
        messages: list[dict | Message],
        system_msgs: list[dict | Message] | None = None,
        stream: bool = True,
        temperature: float | None = None,
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
            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format system and user messages with image support check
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Check if input tokens exceed the per-minute limit and truncate if needed.
            input_tokens = self.count_message_tokens(messages)
            if self.itpm_limit is not None and input_tokens > self.itpm_limit:
                messages = self.truncate_messages(messages)
                input_tokens = self.count_message_tokens(messages)

            # Wait until there is room under the input tokens per minute limit.
            if self.itpm_limit is not None:
                await self.wait_for_token_availability(
                    input_tokens, self.input_token_tracker, self.itpm_limit
                )

            # For output tokens (planned maximum), wait if needed.
            if self.otpm_limit is not None:
                await self.wait_for_token_availability(
                    self.max_tokens, self.output_token_tracker, self.otpm_limit
                )

            # Enforce request-level rate limiting.
            await self.wait_for_request_rate()
            self.last_request_time = time.time()

            params: dict = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_output_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            if not stream:
                # Non-streaming request
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")
                # Update token counts.
                await self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )
                return response.choices[0].message.content

            # Streaming request.
            await self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params, stream=True)
            collected_messages: list[str] = []
            output_text = ""
            async for chunk in response:
                chunk_msg = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_msg)
                output_text += chunk_msg
                print(chunk_msg, end="", flush=True)
            print()  # Newline after streaming.
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # Estimate output token usage.
            output_tokens = self.count_tokens(output_text)
            logger.info(
                f"Estimated output tokens for streaming response: {output_tokens}"
            )
            # Update tracked output tokens.
            await self.update_token_count(0, output_tokens)
            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError:
            logger.exception("Validation error in ask")
            raise
        except OpenAIError as oe:
            logger.exception("OpenAI API error in ask")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception("Unexpected error in ask")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: list[dict | Message],
        images: list[str | dict],
        system_msgs: list[dict | Message] | None = None,
        stream: bool = False,
        temperature: float | None = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
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
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use one of: {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content: list[dict] = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if self.itpm_limit is not None and input_tokens > self.itpm_limit:
                all_messages = self.truncate_messages(all_messages)
                input_tokens = self.count_message_tokens(all_messages)

            if self.itpm_limit is not None:
                await self.wait_for_token_availability(
                    input_tokens, self.input_token_tracker, self.itpm_limit
                )
            if self.otpm_limit is not None:
                await self.wait_for_token_availability(
                    self.max_tokens, self.output_token_tracker, self.otpm_limit
                )

            params: dict = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.model in REASONING_MODELS:
                params["max_output_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Handle non-streaming request
            if not stream:
                response = await self.client.chat.completions.create(**params)
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")
                await self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # Handle streaming request
            await self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)
            collected_messages: list[str] = []
            async for chunk in response:
                chunk_msg = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_msg)
                print(chunk_msg, end="", flush=True)
            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")
            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error in ask_with_images: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
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
        messages: list[dict | Message],
        system_msgs: list[dict | Message] | None = None,
        timeout: int = 300,
        tools: list[dict] | None = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: float | None = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
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

            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))
            input_tokens += tools_tokens

            if self.itpm_limit is not None and input_tokens > self.itpm_limit:
                messages = self.truncate_messages(messages)
                input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            if self.itpm_limit is not None:
                await self.wait_for_token_availability(
                    input_tokens, self.input_token_tracker, self.itpm_limit
                )
            if self.otpm_limit is not None:
                await self.wait_for_token_availability(
                    self.max_tokens, self.output_token_tracker, self.otpm_limit
                )

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with a 'type' field")

            params: dict = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }
            if self.model in REASONING_MODELS:
                params["max_output_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            response: ChatCompletionMessage = await self.client.chat.completions.create(
                **params, stream=False
            )

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print(response)
                return None

            # Update token counts
            await self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )
            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error in ask_tool: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
                logger.error(f"Response: {oe.response.text}")
                logger.error(f"Headers: {oe.response.headers}")
                logger.error(f"params: {params}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise
