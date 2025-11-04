import os
import json
import subprocess
from cog import BasePredictor, Input, Path
from huggingface_hub import snapshot_download

class Predictor(BasePredictor):
    def setup(self):
        # Model ağırlıklarını indir
        os.makedirs("weights", exist_ok=True)
        print("🔽 Downloading model weights...")

        snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="weights/Wan2.1-I2V-14B-480P")
        snapshot_download("TencentGameMate/chinese-wav2vec2-base", local_dir="weights/chinese-wav2vec2-base")
        snapshot_download("MeiGen-AI/InfiniteTalk", local_dir="weights/InfiniteTalk")

    def predict(
        self,
        image: Path = Input(description="Input image or video frame"),
        audio: Path = Input(description="Input audio file (wav or mp3)"),
        mode: str = Input(default="streaming", description="Mode: streaming or clip"),
        size: str = Input(default="infinitetalk-480", description="Resolution"),
    ) -> Path:
        # Dinamik JSON oluştur
        input_json_path = "input.json"
        with open(input_json_path, "w") as f:
            json.dump({
                "image_path": str(image),
                "audio_path": str(audio)
            }, f)

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "result.mp4")

        command = [
            "python", "generate_infinitetalk.py",
            "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", "weights/chinese-wav2vec2-base",
            "--infinitetalk_dir", "weights/InfiniteTalk/single/infinitetalk.safetensors",
            "--input_json", input_json_path,
            "--size", size,
            "--mode", mode,
            "--sample_steps", "40",
            "--motion_frame", "9",
            "--save_file", output_file,
        ]

        subprocess.run(command, check=True)
        return Path(output_file)
