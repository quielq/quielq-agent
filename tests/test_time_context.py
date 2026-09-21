from datetime import datetime

from quielq_agent.time_context import current_time_message


def test_current_time_message_shape():
    message = current_time_message()
    assert message["role"] == "system"
    assert str(datetime.now().year) in message["content"]


def test_current_time_message_respects_timezone_arg():
    manila = current_time_message("Asia/Manila")
    tokyo = current_time_message("Asia/Tokyo")
    assert "Asia/Manila" in manila["content"]
    assert "Asia/Tokyo" in tokyo["content"]
