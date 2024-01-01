import argparse
import logging
import math
import os
import shutil
import time

import PIL.Image
import cv2
import google.generativeai as genai
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pytube import YouTube
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Define the API key
API_KEY = os.environ.get("GEMINI_API")

if API_KEY is None:
    raise ValueError("API_KEY environment variable is not set. Please set it before running the script.")

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define constants for the video URL, file name, output directory, similarity threshold, and sampling interval
DEFAULT_SELECTED_FRAME_DIR = os.path.join(current_dir, 'selected_frames')
SIMILARITY_THRESHOLD = 0.5
SAMPLING_INTERVAL = 10
GEMINI_PRO_VISION = 'gemini-pro-vision'
GEMINI_PRO = 'gemini-pro'
WAIT_TIME = 10
WAIT_TIME_IMG_DESC = 3
MP4_EXTENSION = '.mp4'
DEFAULT_VIDEO_FILENAME = os.path.join(current_dir, 'temp_video.mp4')

# Set up logging
logging.basicConfig(filename=os.path.join(current_dir, 'main.log'), level=logging.INFO)

img_description_prompt = """
Examine the image meticulously and provide a comprehensive description that captures its essence. Ensure your description addresses the following:

- Key Visual Elements: Identify and describe the main objects, people, actions, and settings depicted in the image.
- Relationships and Interactions: Explain how the elements in the image relate to each other and any interactions taking place.
- Context and Significance: If applicable, provide context about the image's source, purpose, or intended audience.
- Emotional Tone and Message: Convey the overall mood or feeling evoked by the image and its intended message.
- Visual Appeal: Highlight any striking visual features or compositional elements that contribute to the image's aesthetic impact.

Describe the image's essence in 30 words, incorporating all of the above aspects.
"""

video_description_prompt = """
Craft a compelling video description that accurately reflects its content, utilizing the provided image descriptions as your guide.
"""

# Define constants for video properties
FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
FRAME_POSITION = cv2.CAP_PROP_POS_FRAMES


def cleanup():
    """Delete the selected frames directory and all mp4 files in the current directory."""
    try:
        # Delete the selected frames directory if it exists
        if os.path.exists(DEFAULT_SELECTED_FRAME_DIR):
            shutil.rmtree(DEFAULT_SELECTED_FRAME_DIR)
            logging.info(f'Deleted {DEFAULT_SELECTED_FRAME_DIR} directory')

        # Delete all mp4 files in the current directory
        for filename in os.listdir(current_dir):
            # Check if the file is an mp4 file
            if filename.endswith(MP4_EXTENSION):
                # Remove the file
                os.remove(os.path.join(current_dir, filename))
                logging.info(f'Deleted {filename}')
    except Exception as e:
        # Log the error and raise it
        logging.error(f'Error occurred while cleaning up: {e}')
        raise


def download_video(url, video_file):
    """Download the video content from the given URL and return a video capture object."""
    try:
        # Create a YouTube object using pytube
        youtube = YouTube(url)
        # Get the highest resolution stream from the YouTube object
        stream = youtube.streams.get_highest_resolution()
        # Download the video to a temporary file
        stream.download(filename=video_file)
        logging.info(f'Downloaded video from {url} to {video_file}')
        # Create and return a video capture object from the temporary file
        video = cv2.VideoCapture(video_file)
        logging.info(f'Created video capture object from {video_file}')
        return video
    except Exception as e:
        # Log the error and raise it
        logging.error(f'Error occurred while downloading video from {url}: {e}')
        raise


def extract_frames(video_capture, output_dir, threshold, interval):
    """Extract the key frames from the video capture object and save them to the output directory."""
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    # Get the total number of frames in the video
    total_frame = video_capture.get(FRAME_COUNT)
    # Initialize an empty list to store the selected frames
    selected_frames = []
    # Initialize a variable to store the previous frame
    prev_frame = None
    # Loop through the frames in the video with a sampling interval
    for frame_idx in tqdm(range(0, int(total_frame), interval), desc="Processing Frames"):
        try:
            # Set the current position of the video capture object
            video_capture.set(FRAME_POSITION, frame_idx)
            # Read the current frame
            success, frame = video_capture.read()
            # Raise an exception if the frame is not valid
            if not success:
                raise ValueError(f'Failed to read frame {frame_idx}')
            # Split the frame into RGB channels
            blue_channel, green_channel, red_channel = cv2.split(frame)
            # If there is a previous frame, compare the similarity with the current frame
            if prev_frame is not None:
                # Compute the SSIM for each channel
                ssim_blue, _ = ssim(prev_frame[0], blue_channel, full=True)
                ssim_green, _ = ssim(prev_frame[1], green_channel, full=True)
                ssim_red, _ = ssim(prev_frame[2], red_channel, full=True)
                # Compute the average SSIM for the frame
                similarity_index = (ssim_blue + ssim_green + ssim_red) / 3
                # If the similarity is below the threshold, add the current frame to the selected frames
                if similarity_index < threshold:
                    selected_frames.append(frame)
                    # Save the selected frame to the output directory
                    frame_filename = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
                    cv2.imwrite(frame_filename, frame)
                    logging.info(f'Saved frame {frame_idx} to {frame_filename}')
            # Update the previous frame
            prev_frame = cv2.split(frame)
        except Exception as e:
            # Log the error and continue the loop
            logging.error(f'Error occurred at frame {frame_idx}: {e}')
            continue
    # Release the video capture object
    video_capture.release()
    # Return the list of selected frames
    return selected_frames


def show_frames(selected_frames):
    """Plot and show the selected frames and save the figure to the output directory."""
    # Calculate the number of rows and columns based on the length of the selected_frames list
    n = math.ceil(math.sqrt(len(selected_frames)))
    # Create a figure and axes for the plot
    fig, axes = plt.subplots(n, n, figsize=(20, 30))
    # Loop through the selected_frames list
    for i, frame in enumerate(selected_frames):
        try:
            # Get the row and column indices from the list index
            row = i // n
            col = i % n
            # Get the subplot object from the axes array
            ax = axes[row, col]
            # Plot the frame on the subplot
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Selected Frame {i}')
            ax.axis('off')
        except Exception as e:
            # Log the error and continue the loop
            logging.error(f'Error occurred at frame {i}: {e}')
            continue

    # Show the plot
    plt.show()


def convert_to_pil_images(selected_frames):
    """Convert the selected frames to PIL images and return a list of them."""
    # Initialize an empty list to store the PIL images
    pil_images = []
    # Loop through the selected frames and convert them to PIL images
    for frame in selected_frames:
        try:
            pil_image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_images.append(pil_image)
            logging.info(f'Converted to PIL image [{pil_image}]')
        except Exception as e:
            # Log the error and continue the loop
            logging.error(f'Error occurred at {frame}: {e}')
            continue

    logging.info(f'Number of converted images to PIL [{len(pil_images)}]')
    print(f'Number of converted images to PIL [{len(pil_images)}]')
    # Return the list of PIL images
    return pil_images


def create_prompt(pil_images, prompt):
    """Create a prompt from the PIL images and the instructions and return a list of them."""
    # Initialize an empty list to store the prompts
    prompts = []
    # Loop through the indices of the pil_images list in increments of 1
    for i in range(0, len(pil_images), 1):
        try:
            # Initialize an empty list to store the prompt
            prompt_list = []
            # Add the prompt
            prompt_list.append(prompt)
            # Extend the list with the 1 images from the pil_images list
            prompt_list.extend(pil_images[i:i + 1])
            # Append the prompt to the prompts list
            prompts.append(prompt_list)
        except Exception as e:
            # Log the error and continue the loop
            logging.error(f'Error occurred at prompt {pil_images[i]}: {e}')
            continue
    # Return the list of prompts
    return prompts


# https://ai.google.dev/tutorials/python_quickstart
def generate_text(prompt, api_key, model_name):
    """Generate text from the prompt using the specified generative model and return the text."""
    # Configure the API key for the generative model
    genai.configure(api_key=api_key)
    # Create a generative model object
    model = genai.GenerativeModel(model_name)
    # Generate text from the prompt
    responses = model.generate_content(prompt)
    # Return the generated text
    return responses


def generate_video_description(pil_images, prompt, api_key):
    """Generate a description of the video using the PIL images and the instructions and return the text."""
    # Initialize an empty list to store the image descriptions
    image_descriptions = []
    # Create prompts from the PIL images and the instructions
    prompts = create_prompt(pil_images, prompt)

    # Generate text from the PIL images using the generative model
    for prompt in prompts:
        try:
            # Generate text from the prompt using the gemini-pro-vision model
            response = generate_text(prompt, api_key, GEMINI_PRO_VISION)
            # Append the generated text to the image descriptions list
            image_descriptions.append(response.text)
            logging.info(
                f'Generated text for prompt [{prompt}] using [{GEMINI_PRO_VISION}] with response --> [{response.text}]')
            time.sleep(WAIT_TIME_IMG_DESC)
        except Exception as e:
            # Log the error and continue the loop
            logging.error(f'Error occurred at prompt {prompt}: {e}')
            continue

    logging.info(f'Described [{len(image_descriptions)}] images from the video.')
    print(f'Described [{len(image_descriptions)}] images from the video.')

    # https://python.langchain.com/docs/use_cases/question_answering/quickstart#indexing-split
    # Join the image descriptions into a single text
    image_text = " ".join(image_descriptions)
    # Split the text into smaller chunks using the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200, add_start_index=True
    )
    split_text = text_splitter.split_text(image_text)

    # Initialize an empty list to store the final prompts
    final_prompts = []
    # Loop through the split text
    for text in split_text:
        # Create a final prompt with the text and a message to wait for the next part
        final_prompts.append(
            f'Do not answer yet. This is just another part of the text I want to send you. Just receive and wait for the next part. {text.strip()}. Remember not answering yet. Just acknowledge you received this part with the message "Waiting for the next part" and wait for the next part.')

    logging.info(f'Number of prompts to be sent to gemini-pro [{len(final_prompts)}]')
    print(f'Number of prompts to be sent to gemini-pro [{len(final_prompts)}]')

    # Create a generative model object for the gemini-pro model
    model = genai.GenerativeModel(GEMINI_PRO)
    # Start a chat with the model
    chat = model.start_chat(history=[])

    for prompt in final_prompts:
        try:
            # Send the prompt to the model and get the response
            response = chat.send_message(prompt)
            # Wait for a few seconds to avoid request timeout
            time.sleep(WAIT_TIME)
            logging.info(f'Sent prompt and received response using {GEMINI_PRO}')
        except Exception as e:
            # Log the error and continue the loop
            logging.error(f'Error occurred at prompt: {e}')
            continue

    # Wait for a few seconds to avoid request timeout
    time.sleep(WAIT_TIME)
    # Send the video description prompt to the model and get the final response
    final_response = chat.send_message(video_description_prompt)
    # Return the final response text
    logging.info(f'Chat history: {chat.history}')
    logging.info(f'Final response: {final_response.text}')
    return final_response.text


if __name__ == '__main__':
    # Create an argument parser object
    parser = argparse.ArgumentParser(description='Download and process a YouTube video')
    # Add an argument for the URL
    parser.add_argument('url', help='The URL of the YouTube video')
    # Add an argument for the selected frame directory with a default value
    parser.add_argument('--selected_frame_dir', help='The directory to save the selected frames',
                        default=DEFAULT_SELECTED_FRAME_DIR)
    # Add an argument for the video file with a default value
    parser.add_argument('--video_file', help='The file name to save the video', default=DEFAULT_VIDEO_FILENAME)
    # Add an argument to show Plot
    parser.add_argument('--show-plot', help='Plot and show the selected frames', action='store_true')
    # Disable cleanup
    parser.add_argument('--no-clean', help='Disable cleanup', action='store_true')
    # Parse the arguments
    args = parser.parse_args()
    # Get the URL, selected frame directory, and video file from the arguments
    url = args.url
    selected_frame_dir = args.selected_frame_dir
    video_file = args.video_file

    # cleanup
    if args.no_clean is False:
        cleanup()

    # Stream the video and get the video capture object
    video_capture = download_video(url, video_file)

    # Extract the key frames from the video capture object
    selected_frames = extract_frames(video_capture, selected_frame_dir, SIMILARITY_THRESHOLD, SAMPLING_INTERVAL)

    # Plot and show the selected frames
    if args.show_plot:
        show_frames(selected_frames)

    # Convert the selected frames to PIL images
    pil_images = convert_to_pil_images(selected_frames)

    # Generate text from the PIL images using the generative model
    response = generate_video_description(pil_images, img_description_prompt, API_KEY)
    print(f'Video Description: {response}')
