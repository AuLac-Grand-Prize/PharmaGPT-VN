from fastapi import FastAPI

from pharmagpt_vn import __version__
from pharmagpt_vn.api.routes import chat, disambiguate, embed, health

app = FastAPI(
    title="PharmaGPT-VN",
    description="LLM tiếng Việt chuyên ngành dược — engine của PharmLink AI.",
    version=__version__,
)

app.include_router(health.router, tags=["health"])
app.include_router(chat.router, prefix="/v1", tags=["chat"])
app.include_router(embed.router, prefix="/v1", tags=["embed"])
app.include_router(disambiguate.router, prefix="/v1", tags=["disambiguate"])
