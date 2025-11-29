from spam_detection.utils.http import ServerCommunication

from spam_detection.models.spam_request import SpamRequest

class SpamService:
    @staticmethod
    def classify_text(text: str) -> bool:
        """Classify text as spam or not spam.

        Args:
            text (str): The text to classify

        Returns:
            bool: True if text is spam, False otherwise
        """
        payload = SpamRequest(text=text)
        response = ServerCommunication.check_spam(payload)
        return response.get("result") == "Spam"
