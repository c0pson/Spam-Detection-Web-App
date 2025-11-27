import requests  # type: ignore
from spam_detection.core.config import SERVER
from spam_detection.models.spam_request import SpamRequest   # <-- import it

class ServerCommunication:

    @staticmethod
    def _handle_response(response):
        if response.status_code != SERVER.AVAILABLE.value:
            raise RuntimeError(f"Server returned {response.status_code} ({response.text})")
        return response.json()

    @staticmethod
    def check_availability() -> bool:
        try:
            response = requests.get(SERVER.URL + SERVER.CHECK_AVAILABILITY, timeout=2)
            data = ServerCommunication._handle_response(response)
            return str(data.get("available")).lower() == "true"
        except requests.ConnectionError as e:
            return False

    @staticmethod
    def check_spam(request: SpamRequest) -> dict:
        """
        Sends a spam check request to the backend using the SpamRequest model.
        """
        response = requests.post(
            SERVER.URL + SERVER.CHECK_SPAM,
            json=request.__dict__,   # convert dataclass to JSON
            timeout=5
        )
        return ServerCommunication._handle_response(response)
