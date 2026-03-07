import faulthandler
import os
import sys
from logging import DEBUG, getLogger
from pathlib import Path
from signal import SIG_DFL, SIGINT, signal
from typing import Annotated

from typer import Argument, Exit, Option, Typer, echo
from vsengine.loops import set_loop

from .app.main import Application, MainWindow
from .app.plugins.manager import PluginManager
from .app.settings import SecretsManager, SettingsManager, ShortcutManager
from .app.settings.models import GlobalSettings
from .assets import load_fonts
from .logging import console, setup_logging
from .vsenv import QtEventLoop

logger = getLogger(__name__)

app = Typer(
    name="vsview",
    help="",
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    add_completion=False,
)


def show_settings_path(value: bool) -> None:
    if value:
        echo(GlobalSettings.path_env)

        raise Exit(0)


def wipe_settings(value: bool) -> None:
    if value:
        GlobalSettings.path_env.unlink(missing_ok=True)
        echo("Global config file sucessfully deleted.")

        raise Exit(0)


def wipe_all_settings(value: bool) -> None:
    if value:
        GlobalSettings.config_path.rmdirs(missing_ok=True, ignore_errors=True)
        echo("Global config path sucessfully deleted.")

        raise Exit(0)


def roaming_settings_callback(value: bool) -> bool:
    if value:
        os.environ["VSVIEW_GLOBAL_SETTINGS_ROAMING"] = "1"

    return value


def env_settings_callback(value: bool) -> bool:
    if value:
        os.environ["VSVIEW_GLOBAL_SETTINGS_ENVIRONMENT"] = "1"

    return value


def env_settings_copy_callback(value: bool) -> bool:
    if value:
        os.environ["VSVIEW_GLOBAL_SETTINGS_ENVIRONMENT_COPY"] = "1"

    return value


input_file_arg = Argument(
    help="Path to input file(s); video(s), image(s) or script(s).",
    metavar="INPUT",
    resolve_path=True,
)

settings_path_opt = Option(
    "--settings-path",
    help=(
        "Print to stdout the resolved [bold]global_settings.json[/bold] path and exit.\n\n"
        "The resolved path respects environment scoping if [bold]--settings-env[/bold] is active.\n\n"
        "Default base directory is [green]%LOCALAPPDATA%\\\vsview\\\\[/green] on Windows, "
        "[green]~/.config/vsview/[/green] on Linux, "
        "and [green]~/Library/Application Support/vsview/[/green] on macOS."
    ),
    is_eager=True,
    callback=show_settings_path,
)
settings_wipe_opt = Option(
    "--settings-wipe",
    help="Delete the [bold]global_settings.json[/bold] file (as shown by [bold]--settings-path[/bold]) and exit.\n\n",
    is_eager=True,
    callback=wipe_settings,
)
settings_wipe_all_opt = Option(
    "--settings-wipe-all",
    help="Delete the entire settings directory (including all environment-scoped subdirectories) and exit.",
    is_eager=True,
    callback=wipe_all_settings,
)
no_settings_opt = Option(
    "--no-settings",
    help="Run without loading or saving any settings for this session.",
)
settings_roaming_opt = Option(
    "--settings-roaming",
    help=(
        "[bold]Windows only[/bold]. Store global settings in [green]%APPDATA%\\\\vsview\\\\[/green] "
        "instead of [green]%LOCALAPPDATA%\\\\vsview\\\\[/green]"
    ),
    envvar="VSVIEW_GLOBAL_SETTINGS_ROAMING",
    is_eager=True,
    callback=roaming_settings_callback,
)
settings_env_opt = Option(
    "--settings-env",
    help="Scope global settings to the active Python environment to prevent conflicts.",
    envvar="VSVIEW_GLOBAL_SETTINGS_ENVIRONMENT",
    is_eager=True,
    callback=env_settings_callback,
)
settings_env_copy_opt = Option(
    "--settings-env-copy",
    help=(
        "If [bold]--settings-env[/bold] is set, and the scoped file doesn't exist yet, "
        "seed it from the base [bold]global_settings.json[/bold]."
    ),
    envvar="VSVIEW_GLOBAL_SETTINGS_ENVIRONMENT_COPY",
    is_eager=True,
    callback=env_settings_copy_callback,
)

verbose_opt = Option(
    "--verbose",
    "-v",
    count=True,
    show_default=False,
    metavar="",
    help="Enable verbose output. Repeat to increase verbosity (-v, -vv, -vvv, ...).",
)


@app.command()
def vsview_cli(
    files: Annotated[list[Path] | None, input_file_arg] = None,
    settings_path: Annotated[bool, settings_path_opt] = False,
    settings_wipe: Annotated[bool, settings_wipe_opt] = False,
    settings_wipe_all: Annotated[bool, settings_wipe_all_opt] = False,
    no_settings: Annotated[bool, no_settings_opt] = False,
    settings_roaming: Annotated[bool, settings_roaming_opt] = False,
    settings_env: Annotated[bool, settings_env_opt] = False,
    settings_env_copy: Annotated[bool, settings_env_copy_opt] = False,
    verbose: Annotated[int, verbose_opt] = 0,
) -> None:
    # Enable faulthandler to get stack traces on segfaults
    faulthandler.enable(file=console.file)

    # Setup env vars
    os.environ["JETPYTOOLS_NO_COLOR"] = "1"
    os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "false"

    # -v -> DEBUG, -vv -> DEBUG - 1, -vvv -> DEBUG - 2, etc.
    setup_logging(level=DEBUG - max(0, verbose - 1) if verbose else None)

    # Set signal handler to default to allow Ctrl+C to work
    signal(SIGINT, SIG_DFL)
    set_loop(QtEventLoop())
    SettingsManager(noop=no_settings)
    ShortcutManager()
    SecretsManager()

    app = Application(sys.argv)

    PluginManager.load()
    load_fonts()

    main_window = MainWindow()
    main_window.ensurePolished()

    if files:
        main_window.show()
        for file in files:
            if file.suffix in [".py", ".vpy"]:
                main_window.load_new_script(file)
            else:
                main_window.load_new_file(file)
    else:
        # Show window first for faster perceived startup
        main_window.show()
        main_window.repaint()
        app.processEvents()

        # Now create default workspaces
        main_window.script_subaction.trigger()
        main_window.file_subaction.trigger()
        main_window.stack.animations_enabled = False
        main_window.quick_script_subaction.trigger()
        main_window.button_group.buttons()[0].click()
        main_window.stack.animations_enabled = True

    sys.exit(app.exec())
