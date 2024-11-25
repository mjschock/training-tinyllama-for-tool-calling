from typing import Dict, List, Literal, Optional

import flet as ft
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# --- FastAPI Models ---
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class Conversation(BaseModel):
    history: List[Message]
    latest_message_options: List[Message]

    def to_dict(self):
        return {
            "history": [msg.to_dict() for msg in self.history],
            "latest_message_options": [
                msg.to_dict() for msg in self.latest_message_options
            ],
        }


# --- FastAPI Backend ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample conversation data
SAMPLE_CONVERSATION = Conversation(
    history=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What's the weather like?"),
        Message(role="assistant", content="I can help you check the weather."),
        Message(role="tool", content="Weather API called: Sunny, 72°F"),
    ],
    latest_message_options=[
        Message(role="assistant", content="It's currently sunny and 72°F."),
        Message(
            role="assistant", content="The temperature is 72°F and the sky is clear."
        ),
    ],
)


@app.get("/conversation")
async def get_conversation():
    return SAMPLE_CONVERSATION.to_dict()


# --- Flet Frontend ---
def main(page: ft.Page):
    page.title = "Conversation Thread"
    page.theme_mode = "light"
    page.padding = 20
    page.scroll = "auto"

    # Set initial window
    # dimensions
    page.window.height = 800
    page.window.width = 600

    def create_message_container(
        message: dict, is_latest: bool = False
    ) -> ft.Container:
        colors = {
            "system": ft.colors.GREY_300,
            "user": ft.colors.BLUE_100,
            "assistant": ft.colors.GREEN_100,
            "tool": ft.colors.ORANGE_100,
        }

        message_container = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text(
                                message["role"].upper(),
                                size=12,
                                weight=ft.FontWeight.BOLD,
                            ),
                            (
                                ft.Text(
                                    "OPTION" if is_latest else "",
                                    size=12,
                                    color=ft.colors.RED_400,
                                    weight=ft.FontWeight.BOLD,
                                )
                                if is_latest
                                else None
                            ),
                        ]
                    ),
                    ft.Text(message["content"]),
                ],
                spacing=5,
            ),
            bgcolor=colors.get(message["role"], ft.colors.WHITE),
            border_radius=10,
            padding=10,
            margin=ft.margin.only(bottom=10),
            width=page.window.width - 40,
        )

        return message_container

    history_column = ft.Column(controls=[], scroll="auto", spacing=10, expand=True)
    options_tabs = ft.Tabs(selected_index=0, animation_duration=300, tabs=[])

    async def update_display():
        # Update history
        history_column.controls.clear()

        for msg in SAMPLE_CONVERSATION.history:
            msg_dict = msg.to_dict()
            container = create_message_container(msg_dict, is_latest=False)

            history_column.controls.append(container)

        # Update options
        options_tabs.tabs.clear()

        for i, msg in enumerate(SAMPLE_CONVERSATION.latest_message_options):
            msg_dict = msg.to_dict()
            container = create_message_container(msg_dict, is_latest=True)
            tab = ft.Tab(text=f"Option {i+1}", content=container)

            options_tabs.tabs.append(tab)

        await page.update_async()

    # Create main sections
    history_section = ft.Column(
        controls=[
            ft.Text("Conversation History", size=24, weight=ft.FontWeight.BOLD),
            history_column,
        ],
        spacing=10,
    )

    options_section = ft.Column(
        controls=[
            ft.Text("Latest Message Options", size=24, weight=ft.FontWeight.BOLD),
            options_tabs,
        ],
        spacing=10,
    )

    # Add sections to page with
    # divider
    page.add(
        history_section, ft.Divider(height=1, color=ft.colors.GREY_400), options_section
    )

    # Set up resize handler
    async def on_resized(e):
        await update_display()

    page.on_resized = on_resized
    page.on_load = update_display


if __name__ == "__main__":
    import threading

    def run_fastapi():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    ft.app(target=main, view=ft.AppView.WEB_BROWSER)
