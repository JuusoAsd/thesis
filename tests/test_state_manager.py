import os
from dotenv import load_dotenv
from src.environments.util import FileManager


load_dotenv()


def test_get_states():
    manager = FileManager("./tests/test_csv", headers=False)
    test_data = [["4", "5", "6"], ["7", "8", "9"], ["1", "2", "3"], ["4", "5", "6"]]
    for i in test_data:
        assert manager.get_next_event() == i


def test_get_states_header():
    manager = FileManager("./tests/test_csv", headers=True)
    test_data = [["7", "8", "9"], ["4", "5", "6"]]
    for i in test_data:
        assert manager.get_next_event() == i


def test_last_file():
    manager = FileManager("./tests/test_csv", headers=False)
    for i in range(5):
        manager.get_next_event()

    for i in range(3):
        assert manager.get_next_event() is None


def test_get_single_file():
    manager = FileManager("./tests/test_csv/test1.txt", headers=False)
    test_data = [["4", "5", "6"], ["7", "8", "9"]]
    for i in test_data:
        assert manager.get_next_event() == i

    assert manager.get_next_event() is None
