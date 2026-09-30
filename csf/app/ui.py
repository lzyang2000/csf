# SPDX-License-Identifier: Apache-2.0
"""Small viser helpers shared by the panels: toasts and spinners."""
from __future__ import annotations


def toast_error(client: object, title: str, body: str, *, seconds: float = 10.0) -> None:
    """A red, auto-closing error toast."""
    client.add_notification(title=title, body=body, color="red", auto_close_seconds=seconds)


def notify(app: object, session: object, title: str, body: str, color: str | None = None,
           *, seconds: float = 6.0) -> None:
    """A toast that is safe to call from any thread, and never raises."""
    if not app.client_active(session.client_id):
        return
    try:
        kwargs = {"title": title, "body": body, "auto_close_seconds": seconds}
        if color is not None:
            kwargs["color"] = color
        session.client.add_notification(**kwargs)
    except Exception:  # noqa: BLE001 - a toast is not worth an exception
        pass


def busy(client: object, title: str, body: str):
    """A spinner; resolve it with `finish` or `fail`."""
    return client.add_notification(title=title, body=body, loading=True, with_close_button=False)


def finish(notification: object, title: str | None = None, body: str | None = None,
           *, seconds: float = 5.0) -> None:
    """Resolve a `busy` spinner green."""
    if title is not None:
        notification.title = title
    if body is not None:
        notification.body = body
    notification.loading = False
    notification.with_close_button = True
    notification.auto_close_seconds = seconds
    notification.color = "green"


def fail(notification: object) -> None:
    """Stop a `busy` spinner (pair with `toast_error`)."""
    notification.loading = False
    notification.with_close_button = True
