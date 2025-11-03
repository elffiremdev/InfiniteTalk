from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def predict(self, text: str = Input(description="Metni geri döndür", default="Merhaba Cog!")) -> str:
        return text
