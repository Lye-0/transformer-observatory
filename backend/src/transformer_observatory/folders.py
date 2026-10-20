import os
import shutil
import subprocess


def open_folder(path):
    """Called only for server-selected storage paths after an explicit UI action."""
    if os.name == "nt":
        os.startfile(str(path))
        return
    explorer = shutil.which("explorer.exe") if os.environ.get("WSL_DISTRO_NAME") else None
    if explorer:
        converted = subprocess.run(
            ["wslpath", "-w", str(path)], check=True, capture_output=True, text=True
        ).stdout.strip()
        command = [explorer, converted]
    elif opener := shutil.which("xdg-open"):
        command = [opener, str(path)]
    else:
        raise ValueError("ファイルマネージャーが見つかりません。表示された保存先を手動で開いてください")
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
