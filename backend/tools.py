"""
Tools for the Simple Agent
- Bash execution
- File operations (read, write, list, search)
- Skill system (find, install, load, execute)
"""

import os
import subprocess
import tempfile
from pathlib import Path
import platform
from typing import Optional

from langchain.tools import tool


# =============================================================================
# Configuration
# =============================================================================

def get_skills_dir() -> Path:
    """Get the skills directory (create if not exists)"""
    script_dir = Path(__file__).parent
    skills_dir = script_dir / ".agents" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    return skills_dir


# =============================================================================
# Bash Execution Tools
# =============================================================================

@tool
def run_command(description: str, command: str) -> str:
    """
    Execute a shell command and return the output.

    IMPORTANT SYSTEM INFO:
    Current OS is {platform.system()}.
    If this is Windows, YOU MUST use Windows cmd.exe commands (e.g., 'dir' instead of 'ls', 'type' instead of 'cat').
    If this is Linux/Mac, use standard bash commands.

    SECURITY: This tool only allows read operations within the current working directory.
    Any attempts to access files outside the working directory will be rejected.

    Args:
        description: Brief description of why you're running this command
        command: The shell command to execute

    Returns:
        Command output (stdout + stderr)
    """
    # Security: Define allowed working directory
    allowed_cwd = Path(__file__).parent.resolve()

    # Security: Block dangerous patterns that could escape the working directory
    dangerous_patterns = [
        # Directory traversal
        '..',
        # Absolute paths (Windows and Unix)
        ':\\',  # Windows drive paths like C:\
        '/etc/', '/home/', '/root/', '/tmp/', '/var/',  # Unix system dirs
        'C:\\', 'D:\\', 'E:\\',  # Windows drive letters
        # Dangerous commands that could modify system state
        'rm -rf /', 'rm -rf /*',
        'format', 'diskpart',  # Windows disk commands
        'chmod 777', 'chown ', 'sudo ', 'su ',
        # Environment/network attacks
        # 'curl ', 'wget ', 'nc ', 'netcat ',
        # 'ssh ', 'scp ', 'ftp ',
        # Data exfiltration
        'base64 ', 'xxd ', 'od ',
        # # Process manipulation
        # 'kill ', 'pkill ', 'killall ',
        # Privilege escalation
        'setuid', 'setgid', 'passwd ', 'useradd ', 'usermod ',
    ]

    cmd_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in cmd_lower:
            return f"Error: Command blocked for security reasons - contains restricted pattern: '{pattern}'"

    # Security: Check for absolute paths in common file operations
    # Block commands that try to read files outside cwd
    read_commands = ['cat ', 'type ', 'head ', 'tail ', 'less ', 'more ', 'grep ']
    for cmd in read_commands:
        if cmd in cmd_lower:
            # Extract the path argument
            parts = command.split(cmd, 1)[-1].strip().split()
            for part in parts:
                # Skip flags
                if part.startswith('-'):
                    continue
                # Check if it's an absolute path or contains ..
                if part.startswith('/') or ':' in part or '..' in part:
                    return f"Error: Command blocked - cannot access files outside working directory: '{part}'"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8", # 强制使用 utf-8 解码输出
            timeout=300,
            cwd=str(allowed_cwd)
        )
        output = result.stdout
        if result.stderr:
            output += "\nStderr:\n" + result.stderr
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (5 minutes)"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# File Operation Tools
# =============================================================================

@tool
def read_file(description: str, path: str) -> str:
    """
    Read the contents of a file.

    Args:
        description: Brief description of why you're reading this file
        path: Absolute or relative path to the file

    Returns:
        File contents
    """
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(__file__).parent / path

        if not file_path.exists():
            return f"Error: File not found: {path}"

        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(description: str, path: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        description: Brief description of what you're writing
        path: Path to the file
        content: Content to write

    Returns:
        Success message
    """
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(__file__).parent / path

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def list_directory(description: str, path: str = ".") -> str:
    """
    List contents of a directory.

    Args:
        description: Brief description of what you're listing
        path: Directory path (default: current directory)

    Returns:
        Directory contents as a tree
    """
    try:
        dir_path = Path(path)
        if not dir_path.is_absolute():
            dir_path = Path(__file__).parent / path

        if not dir_path.exists():
            return f"Error: Directory not found: {path}"

        lines = [f"Contents of {dir_path}:"]
        for item in sorted(dir_path.iterdir()):
            prefix = "[DIR] " if item.is_dir() else ""
            lines.append(f"  {prefix}{item.name}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool
def find_files(description: str, pattern: str = "*.py", path: str = ".") -> str:
    """
    Find files matching a glob pattern.

    Args:
        description: Brief description of what you're searching for
        pattern: Glob pattern (e.g., "*.py", "**/*.txt")
        path: Base directory to search

    Returns:
        List of matching files
    """
    try:
        base_path = Path(path)
        if not base_path.is_absolute():
            base_path = Path(__file__).parent / path

        if not base_path.exists():
            return f"Error: Directory not found: {path}"

        # Handle ** patterns for recursive search
        if "**" in pattern:
            matches = list(base_path.rglob(pattern.replace("**/", "")))
        else:
            matches = list(base_path.glob(pattern))

        if not matches:
            return f"No files found matching '{pattern}' in {path}"

        lines = [f"Found {len(matches)} file(s) matching '{pattern}':"]
        for match in sorted(matches):
            rel_path = match.relative_to(base_path)
            lines.append(f"  {rel_path}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Skill System Tools
# =============================================================================

@tool
def list_skills(description: str = "List available skills") -> str:
    """
    List all available skills (both built-in and custom).

    Args:
        description: Brief description (optional)

    Returns:
        List of skill names and descriptions
    """
    skills_dir = get_skills_dir()

    if not skills_dir.exists():
        return "No skills directory found."

    skills = []
    for skill_dir in skills_dir.iterdir():
        if skill_dir.is_dir() and not skill_dir.name.startswith("."):
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                content = skill_md.read_text(encoding="utf-8")
                # Parse frontmatter
                name = skill_dir.name
                desc = "No description"

                if content.startswith("---"):
                    lines = content.split("\n")
                    for line in lines[1:]:
                        if line.startswith("---"):
                            break
                        if line.startswith("description:"):
                            desc = line.replace("description:", "").strip()
                            break

                skills.append(f"- **{name}**: {desc}")

    if not skills:
        return "No skills found."

    return "\n".join(skills)


@tool
def get_skill(description: str, skill_name: str) -> str:
    """
    Get the full content of a skill.

    Args:
        description: Brief description of why you're getting this skill
        skill_name: Name of the skill to retrieve

    Returns:
        Full SKILL.md content
    """
    skills_dir = get_skills_dir()
    skill_path = skills_dir / skill_name / "SKILL.md"

    if not skill_path.exists():
        return f"Error: Skill '{skill_name}' not found."

    return skill_path.read_text(encoding="utf-8")






@tool
def execute_skill_script(description: str, skill_name: str, script_name: str, script_args: str = "") -> str:
    """
    Execute a script within a skill.

    Args:
        description: Brief description of what you're doing
        skill_name: Name of the skill
        script_name: Script filename (e.g., "script.py" or "script.sh")
        script_args: Optional command-line arguments (separated by spaces)

    Returns:
        Script output
    """
    skills_dir = get_skills_dir()
    script_path = skills_dir / skill_name / "scripts" / script_name

    if not script_path.exists():
        return f"Error: Script '{script_name}' not found in skill '{skill_name}'"

    # Build command (注意这里改成了 script_args)
    if script_name.endswith(".py"):
        cmd = f"python \"{script_path}\" {script_args}"
    elif script_name.endswith(".sh"):
        # 如果是 Windows，跑 .sh 可能会失败，除非装了 Git Bash
        cmd = f"bash \"{script_path}\" {script_args}"
    elif script_name.endswith(".js"):
        cmd = f"node \"{script_path}\" {script_args}"
    else:
        return f"Error: Unknown script type: {script_name}"

    # 准备环境变量，强制 Python 子进程使用 UTF-8 输出，解决 GBK 报错
    custom_env = os.environ.copy()
    custom_env["PYTHONIOENCODING"] = "utf-8"

    # Execute
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8", # 告诉 Python 主进程用 UTF-8 读取 stdout
            env=custom_env,   # 注入环境变量
            timeout=300,
            cwd=str(skills_dir / skill_name)
        )
        output = result.stdout
        if result.stderr:
            output += "\nStderr:\n" + result.stderr
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Script timed out"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Tool Collection
# =============================================================================

def get_all_tools():
    """Get all available tools."""
    return [
        # Bash
        run_command,
        # File operations
        read_file,
        write_file,
        list_directory,
        find_files,
        # Skill system
        list_skills,
        get_skill,
        # install_skill,
        # search_skills_online,
        execute_skill_script,
    ]
