from spam_detection.utils.http import ServerCommunication

from spam_detection.models.spam_request import SpamRequest

class SpamService:
    @staticmethod
    def classify_text(text: str) -> bool:
        payload = SpamRequest(text=text)
        response = ServerCommunication.check_spam(payload)
        return response.get("result") == "Spam"
