from enum import Enum

class SERVER(Enum):
    URL = "http://79.76.44.20"
    CHECK_SPAM = "/check_spam"
    CHECK_AVAILABILITY = "/check_availability"
    AVAILABLE = 200

    def __add__(self, other):
        return self.value + other.value
