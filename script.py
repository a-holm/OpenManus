import asyncio
import argparse
import sys

from app.agent.manus import Manus
from app.logger import logger


async def process_prompt(agent, prompt):
    """Process a single prompt with error handling."""
    if not prompt.strip():
        logger.warning("Empty prompt skipped.")
        return False

    try:
        logger.warning(f"Processing prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        # Add a timeout to prevent hanging (e.g., 10 minutes)
        await asyncio.wait_for(agent.run(prompt), timeout=3600)
        logger.info("Request processing completed successfully.")
        return True
    except asyncio.TimeoutError:
        logger.error("Processing prompt timed out after 1 hour.")
        return False
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}")
        return False


async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the Manus agent with prompts.')
    parser.add_argument('prompt', nargs='?', default=None, help='The prompt to process')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('-c', '--continuous', action='store_true', help='Run in continuous mode')
    parser.add_argument('-f', '--file', help='Path to a file containing prompts (one per line)')

    # Parse arguments
    args = parser.parse_args()

    agent = Manus()

    try:
        # Process prompts from file
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as file:
                    total_prompts = 0
                    successful_prompts = 0

                    for line_num, line in enumerate(file, 1):
                        prompt = line.strip()
                        if prompt:
                            total_prompts += 1
                            if await process_prompt(agent, prompt):
                                successful_prompts += 1

                    logger.info(f"File processing complete. {successful_prompts}/{total_prompts} prompts processed successfully.")
                return
            except FileNotFoundError:
                logger.error(f"File not found: {args.file}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                sys.exit(1)

        # Interactive or single prompt mode
        continuous_mode = args.continuous or args.interactive

        while True:
            # Get prompt from command line arguments or interactively
            if args.interactive or args.prompt is None:
                try:
                    prompt = input("Enter your prompt (or 'exit' to quit): ")
                    if prompt.lower() == 'exit':
                        break
                except EOFError:
                    logger.info("Input stream ended. Exiting.")
                    break
            else:
                prompt = args.prompt

            await process_prompt(agent, prompt)

            # If not in continuous mode and prompt was from command line, exit after one execution
            if not continuous_mode:
                break

            # Reset prompt if it was from command line so next iteration uses interactive input
            args.prompt = None

    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")


if __name__ == "__main__":
    asyncio.run(main())
