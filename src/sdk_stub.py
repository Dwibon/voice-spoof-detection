import requests

class VoiceSpoofDetectorClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def detect(self, audio_path: str):
        """
        Detect spoofing on a single audio file.
        """
        url = f"{self.base_url}/detect"
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()

    def detect_with_reference(self, audio_path: str, reference_path: str):
        """
        Detect spoofing with an enrolled speaker reference.
        """
        url = f"{self.base_url}/detect-with-reference"
        with open(audio_path, "rb") as faudio, open(reference_path, "rb") as fref:
            files = {
                "audio": faudio,
                "reference": fref
            }
            response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()

    def check_health(self):
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    client = VoiceSpoofDetectorClient()
    print("Health Check:", client.check_health())
