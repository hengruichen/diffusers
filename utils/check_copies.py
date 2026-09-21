import os
import re
import subprocess
from pathlib import Path

from tqdm import tqdm

from utils.utils import get_all_files, get_files_with_suffix, get_files_with_suffix_and_dir


def check_copies():
    """Check if there are any copies of files in the repository."""
    # Get all files in the repository
    files = get_all_files(Path.cwd())

    # Get all files with .py extension
    files_with_suffix = get_files_with_suffix(files, ".py")

    # Get all files with .py and .ipynb extensions
    files_with_suffix_and_dir = get_files_with_suffix_and_dir(files, ".py", ".ipynb")

    # Check for copies of files
    for file in tqdm(files_with_suffix_and_dir, desc="Checking for copies of files"):
        # Get the file name and directory
        file_name = file.name
        file_dir = file.parent

        # Check for copies of the file in the same directory
        for other_file in files_with_suffix_and_dir:
            if other_file.name == file_name and other_file.parent == file_dir:
                # Get the path to the other file
                other_file_path = other_file.relative_to(Path.cwd())

                # Get the path to the file
                file_path = file.relative_to(Path.cwd())

                # Check if the other file is a copy of the file
                if other_file_path != file_path:
                    # Get the path to the other file
                    other_file_path = other_file.relative_to(Path.cwd())

                    # Get the path to the file
                    file_path = file.relative_to(Path.cwd())

                    # Print the message
                    print(f"Found a copy of {file_path} in {other_file_path}")


def check_copies2():
    """Check if there are any copies of files in the repository."""
    # Get all files in the repository
    files = get_all_files(Path.cwd())

    # Get all files with .py extension
    files_with_suffix = get_files_with_suffix(files, ".py")

    # Get all files with .py and .ipynb extensions
    files_with_suffix_and_dir = get_files_with_suffix_and_dir(files, ".py", ".ipynb")

    # Check for copies of files
    for file in tqdm(files_with_suffix_and_dir, desc="Checking for copies of files"):
        # Get the file name and directory
        file_name = file.name
        file_dir = file.parent

        # Check for copies of the file in the same directory
        for other_file in files_with_suffix_and_dir:
            if other_file.name == file_name and other_file.parent == file_dir:
                # Get the path to the other file
                other_file_path = other_file.relative_to(Path.cwd())

                # Get the path to the file
                file_path = file.relative_to(Path.cwd())

                # Check if the other file is a copy of the file
                if other_file_path != file_path:
                    # Get the path to the other file
                    other_file_path = other_file.relative_to(Path.cwd())

                    # Get the path to the file
                    file_path = file.relative_to(Path.cwd())

                    # Print the message
                    print(f"Found a copy of {file_path} in {other_file_path}")


def check_copies3():
    """Check if there are any copies of files in the repository."""
    # Get all files in the repository
    files = get_all_files(Path.cwd())

    # Get all files with .py extension
    files_with_suffix = get_files_with_suffix(files, ".py")

    # Get all files with .py and .ipynb extensions
    files_with_suffix_and_dir = get_files_with_suffix_and_dir(files, ".py", ".ipynb")

    # Check for copies of files
    for file in tqdm(files_with_suffix_and_dir, desc="Checking for copies of files"):
        # Get the file name and directory
        file_name = file.name
        file_dir = file.parent

        # Check for copies of the file in the same directory
        for other_file in files_with_suffix_and_dir:
            if other_file.name == file_name and other_file.parent == file_dir:
                # Get the path to the other file
                other_file_path = other_file.relative_to(Path.cwd())

                # Get the path to the file
                file_path = file.relative_to(Path.cwd())

                # Check if the other file is a copy of the file
                if other_file_path != file_path:
                    # Get the path to the other file
                    other_file_path = other_file.relative_to(Path.cwd())

                    # Get the path to the file
                    file_path = file.relative_to(Path.cwd())

                    # Print the message
                    print(f"Found a copy of {file_path} in {other_file_path}")


def check_copies4():
    """Check if there are any copies of files in the repository."""
    # Get all files in the repository
    files = get_all_files(Path.cwd())

    # Get all files with .py extension
    files_with_suffix = get_files_with_suffix(files, ".py")

    # Get all files with .py and .ipynb extensions
    files_with_suffix_and_dir = get_files_with_suffix_and_dir(files, ".py", ".ipynb")

    # Check for copies of files
    for file in tqdm(files_with_suffix_and_dir, desc="Checking for copies of files"):
        # Get the file name and directory
        file_name = file.name
        file_dir = file.parent

        # Check for copies of the file in the same directory
        for other_file in files_with_suffix_and_dir:
            if other_file.name == file_name and other_file.parent == file_dir:
                # Get the path to the other file
                other_file_path = other_file.relative_to(Path.cwd())

                # Get the path to the file
                file_path = file.relative_to(Path.cwd())

                # Check if the other file is a copy of the file
                if other_file_path != file_path:
                    # Get the path to the other file
                    other_file_path = other_file.relative_to(Path.cwd())

                    # Get the path to the file
                    file_path = file.relative_to(Path.cwd())

                    # Print the message
                    print(f"Found a copy of {file_path} in {other_file_path}")


def check_copies5():
    """Check if there are any copies of files in the repository."""
    # Get all files in the repository
    files = get_all_files(Path.cwd())

    # Get all files with .py extension
    files_with_suffix = get_files_with_suffix(files, ".py")

    # Get all files with .py and .ipynb extensions
    files_with_suffix_and_dir = get_files_with_suffix_and_dir(files, ".py", ".ipynb")

    # Check for copies of files
    for file in tqdm(files_with_suffix_and_dir, desc="Checking for copies of files"):
        # Get the file name and directory
        file_name = file.name
        file_dir = file.parent

        # Check for copies of the file in the same directory
        for other_file in files_with_suffix_and_dir:
            if other_file.name == file_name and other_file.parent == file_dir:
                # Get the path to the other file
                other_file_path = other_file.relative_to(Path.cwd())

                # Get the path to the file
                file_path = file.relative_to(Path.cwd())

                # Check if the other file is a copy of the file
                if other_file_path != file_path:
                    # Get the path to the other file
                    other_file_path = other_file.relative_to(Path.cwd())

                    # Get the path to the file
                    file_path = file.relative_to(Path.cwd())

                    # Print the message
                    print(f"Found a copy of {file_path} in {other_file_path}")


def check_copies6():
    """Check if there are any copies of files in the repository."""
    # Get all files in the repository
    files = get_all_files(Path.cwd())

    # Get all files with .py extension
    files_with_suffix = get_files_with_suffix(files, ".py")

    # Get all files with .py and .ipynb extensions
    files_with_suffix_and_dir = get_files_with_suffix_and_dir(files, ".py", ".ipynb")

    # Check for copies of files
    for file in tqdm(files_with_suffix_and_dir, desc="Checking for copies of files"):
        # Get the file name and directory
        file_name = file.name
        file_dir = file.parent

        # Check for copies of the file in the same directory
        for other_file in files_with_suffix_and_dir:
            if other_file.name == file_name and other_file.parent == file_dir:
                # Get the path to the other file
                other_file_path = other_file.relative_to(Path.cwd())

                # Get the path to the file
                file_path = file.relative_to(Path.cwd())

                # Check if the other file is a copy of the file
                if other_file_path != file_path:
                    # Get the path to the other file
                    other_file_path = other_file.relative_to(Path.cwd())

                    # Get the path to the file
                    file_path = file.relative_to(Path.cwd())

                    # Print the message
                    print(f"Found a copy of {file_path} in {other_file_path}")


def check_copies7():
    """Check if there are any copies of files in the repository."""
    # Get all files in the repository
    files = get_all_files(Path.cwd())

    # Get all files with .py extension
    files_with_suffix = get_files_with_suffix(files, ".py")

    # Get all files with .py and .ipynb extensions
    files_with_suffix_and_dir = get_files_with_suffix_and_dir(files, ".py", ".ipynb")

    # Check for copies of files
    for file in tqdm(files_with_suffix