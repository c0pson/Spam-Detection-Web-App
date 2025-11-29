from enum import Enum

class SERVER(Enum):
    URL = "http://79.76.44.20"
    CHECK_SPAM = "/check_spam"
    CHECK_AVAILABILITY = "/check_availability"
    AVAILABLE = 200

    def __add__(self, other):
        """Returns the sum of the values of two SERVER enum members.

        Args:
            other (SERVER): Another SERVER enum member to add.

        Returns:
            str: The combined value of the two enum members.
        """
        return self.value + other.value
