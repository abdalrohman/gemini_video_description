# Gemini Video Description
Gemini Video Description is a Python project that uses the Google Generative AI and the Gemini models to generate a description of a YouTube video based on the key frames extracted from the video. The project is in early stage and the code is inspired from this [Kaggle notebook](https://www.kaggle.com/code/ashishkumarak/gemini-api-video-prompting).

## Export GEMINI_API Key

Using Linux
```bash
export GEMINI_API=your_api_key
```

Using Powershell
```bash
$env:GEMINI_API = "your_api_key"
```

## Usage

To use the project, you need to have a valid API key for the Google Generative AI. You can get one from here.

You also need to install the required dependencies from the `requirements.txt` file. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

Then, you can run the main script `main.py` with the URL of the YouTube video as a command line argument. For example:

```python
python main.py "https://www.youtube.com/watch?v=example"
```

You can also specify the directory to save the selected frames and the file name to save the video as optional arguments. For example:

```python
python main.py https://www.youtube.com/watch?v=example --selected_frame_dir my_frames --video_file my_video.mp4
```

If you omit these arguments, the script will use the default values `selected_frames` and `temp_video.mp4`.

The script will download the video, extract the key frames, generate text for each frame, and then generate a final description of the video using the Gemini models. The script will also plot and show the selected frames and save the figure to the output directory.

## Requirements
The project requires the following Python modules:

- google-generativeai==0.3.2
- matplotlib
- mpld3
- numpy
- opencv-python==4.9.0.80
- Pillow
- scikit-image==0.22.0
- tqdm==4.66.1

You can install them using the `requirements.txt` file as mentioned above.

## TODO
The project is still under development and has the following TODO items:

- [ ] Optimize Frame Description Prompt
- [ ] Extend Prompt Functionality for Universal Video Descriptions
- [ ] Implement Chat History Archival
- [ ] Integrate Configuration File Reader
- [ ] Implement Enhanced Logging with Console Output
