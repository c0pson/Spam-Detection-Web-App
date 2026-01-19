import os
import requests
from spam_detection.core.config import SERVER, MODEL_DIR
from spam_detection.artifacts.models import SpamRequest
from requests.exceptions import ConnectionError, ReadTimeout

class ServerCommunication:
    """Class to handle communication with the server for spam detection.
    Provides methods to check the availability of the server and to send
    spam check requests. It handles HTTP responses and raises errors for
    non-success status codes.
    """
    @staticmethod
    def _handle_response(response: requests.Response):
        """Handles the HTTP response from the server.

        Args:
            response (requests.Response): The HTTP response object from the server.

        Raises:
            RuntimeError: If the server returns an error status code.

        Returns:
            dict: The JSON response from the server.
        """
        if response.status_code != SERVER.AVAILABLE.value:
            raise RuntimeError(f"Server returned {response.status_code} ({response.text})")
        return response.json()

    @staticmethod
    def check_availability() -> bool:
        """Check if the server is available.

        Returns:
            bool: True if server is available, False otherwise.
        """
        if os.path.isdir(MODEL_DIR):
            return True
        try:
            response = requests.get(
                SERVER.URL + SERVER.CHECK_AVAILABILITY,
                timeout=5
            )
            data = ServerCommunication._handle_response(response)
            return str(data.get("available")).lower() == "true"
        except (ConnectionError, ReadTimeout) as e:
            return False
        except Exception as e:
            return False

    @staticmethod
    def check_spam(request: SpamRequest) -> dict:
        """Sends a spam check request to the backend using the SpamRequest model.

        Args:
            request (SpamRequest): The spam request object containing the data to be checked.

        Returns:
            dict: The JSON response from the server indicating the spam check result.
        """
        response = requests.post(
            SERVER.URL + SERVER.CHECK_SPAM,
            json=request.__dict__,
            timeout=5
        )
        return ServerCommunication._handle_response(response)
