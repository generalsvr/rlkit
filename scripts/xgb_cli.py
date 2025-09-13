from __future__ import annotations

import typer

from rl_lib.xgb.topbot import topbot_train, topbot_eval
from rl_lib.xgb.logret import logret_train, logret_eval
from rl_lib.xgb.trendchange import trendchange_train, trendchange_eval
from rl_lib.xgb.impulse import impulse_train, impulse_eval


app = typer.Typer(add_completion=False)


# Register commands under the same names as the legacy monolith
app.command("topbot-train")(topbot_train)
app.command("topbot-eval")(topbot_eval)
app.command("logret-train")(logret_train)
app.command("logret-eval")(logret_eval)
app.command("trendchange-train")(trendchange_train)
app.command("trendchange-eval")(trendchange_eval)
app.command("impulse-train")(impulse_train)
app.command("impulse-eval")(impulse_eval)


if __name__ == "__main__":
    app()


