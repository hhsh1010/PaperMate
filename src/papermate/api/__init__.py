"""PaperMate · API 包

暴露 `app` 单例，使两种启动方式等价：
- `python -m papermate.api`
- `uvicorn papermate.api:app`
"""

from .app import create_app

app = create_app()
