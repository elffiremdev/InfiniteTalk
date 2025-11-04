import os
from cog import BasePredictor, Input, Path
import subprocess

class Predictor(BasePredictor):
    def setup(self):
        # model klasörlerini oluştur
        os.makedirs("weights", exist_ok=True)

    def predict(
        self,
        image: Path = Input(description="Input image or video frame"),
        audio: Path = Input(description="Input audio file (wav or mp3)"),
        mode: str = Input(default="streaming", description="Generation mode: streaming or clip"),
        size: str = Input(default="infinitetalk-480", description="Video size: infinitetalk-480 or infinitetalk-720"),
    ) -> Path:
        """
        InfiniteTalk video generation using pre-downloaded weights
        """
        output_file = "output.mp4"

        command = [
            "python", "generate_infinitetalk.py",
            "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", "weights/chinese-wav2vec2-base",
            "--infinitetalk_dir", "weights/InfiniteTalk/single/infinitetalk.safetensors",
            "--input_json", "examples/single_example_image.json",
            "--size", size,
            "--mode", mode,
            "--sample_steps", "40",
            "--motion_frame", "9",
            "--save_file", output_file,
        ]

        subprocess.run(command, check=True)

        return Path(output_file)
