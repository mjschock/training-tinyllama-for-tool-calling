# main.py
import flet as ft
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Literal, Optional
from pydantic import BaseModel
import uvicorn


# --- FastAPI Models ---
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class Conversation(BaseModel):
    history: List[Message]
    latest_message_options: List[Message]  # List of possible latest messages


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
    return SAMPLE_CONVERSATION


# --- Flet Frontend ---
def main(page: ft.Page):
    page.title = "Conversation Thread"
    page.theme_mode = "light"
    page.padding = 20

    history_container = ft.Column(
        spacing=10,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )

    latest_options_container = ft.Container(
        padding=ft.padding.only(top=20),
    )

    def create_message_container(
        message: dict, is_latest: bool = False
    ) -> ft.Container:
        role = message["role"]
        content = message["content"]

        # Define colors and alignment based on role
        colors = {
            "system": ft.colors.GREY_300,
            "user": ft.colors.BLUE_100,
            "assistant": ft.colors.GREEN_100,
            "tool": ft.colors.ORANGE_100,
        }
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(role.upper(), size=12, weight=ft.FontWeight.BOLD),
                            (
                                ft.Text(
                                    "OPTION" if is_latest else "",
                                    size=12,
                                    color=ft.colors.RED_400,
                                    weight=ft.FontWeight.BOLD,
                                )
                                if is_latest
                                else ft.Container()
                            ),
                        ]
                    ),
                    ft.Text(content),
                ]
            ),
            bgcolor=colors.get(role, ft.colors.WHITE),
            padding=10,
            border_radius=8,
            width=page.window_width - 40,
        )

    def create_latest_options_view(options: List[dict]) -> ft.Container:
        tabs = ft.Tabs(
            selected_index=0,
            tabs=[
                ft.Tab(
                    text=f"Option {i+1}",
                    content=create_message_container(option, is_latest=True),
                )
                for i, option in enumerate(options)
            ],
        )
        return ft.Container(content=tabs)

    async def load_conversation():
        # Clear existing messages
        history_container.controls.clear()

        # In a real app, this would fetch from the
        # FastAPI endpoint Load history messages
        for message in SAMPLE_CONVERSATION.history:
            message_dict = message.dict()
            history_container.controls.append(
                create_message_container(message_dict, is_latest=False)
            )

        # Load latest message options
        latest_options = [
            msg.dict() for msg in SAMPLE_CONVERSATION.latest_message_options
        ]
        latest_options_container.content = create_latest_options_view(latest_options)

        await page.update_async()

    # Create divider between history and latest
    # message
    divider = ft.Divider(height=1, color=ft.colors.GREY_400)
    page.add(
        ft.Text("Conversation History", size=24, weight=ft.FontWeight.BOLD),
        history_container,
        divider,
        ft.Text("Latest Message Options", size=24, weight=ft.FontWeight.BOLD),
        latest_options_container,
    )

    page.window_width = 600
    page.window_height = 800

    # Load initial conversation
    page.on_load = load_conversation


if __name__ == "__main__":
    # Run both servers
    import threading

    # Start FastAPI server in a separate thread
    def run_fastapi():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Start Flet app
    ft.app(target=main, view=ft.AppView.WEB_BROWSER)
