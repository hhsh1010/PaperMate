"""PaperMate · 应用配置（pydantic-settings，密钥走环境变量不入库，docs/10）"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PAPERMATE_",
        env_file=".env",
        extra="ignore",
    )

    # 数据目录：相对路径一律基于项目根解析（非进程 CWD），杜绝
    # 「从 web/ 等子目录启动后端 → 生成 web/data/ 空库」的分叉（2026-08-22 修复）。
    # 仍可用 PAPERMATE_DATA_DIR 环境变量显式覆盖为任意绝对路径。
    data_dir: str = "data"
    database_url: str = ""
    # 服务监听（2026-09-06）：此前 .env 里的 PAPERMATE_HOST/PORT 是**死配置**（端口硬编码在
    # __main__.py），用户改了不生效，容易在「以为已限制访问」的情况下暴露服务。
    # 默认只监听回环——本项目定位为本地单用户、无鉴权，绑 0.0.0.0 会把局域网内的全部
    # 文档/记忆/设置（含 PUT /settings 覆盖密钥）暴露出去。需局域网访问请显式设
    # PAPERMATE_HOST=0.0.0.0 并自行评估风险。
    host: str = "127.0.0.1"
    port: int = 8000
    default_model: str = "fake"  # fake | deepseek | openai | qwen
    default_embedder: str = "real"  # fake | real(BGE-M3)
    # BGE-M3 稀疏向量增强（与 BM25 并行的词法信号，同向量空间融合更一致）。
    # 已验证真实 BGE-M3 在 GPU 上可加载、稠密/稀疏编码正常（dim=1024, sparse 非空），
    # 设置页默认启用，后端与 UI 对齐为默认开启；测试环境通过 env 回退 fake 避免加载权重。
    use_bge_sparse: bool = True  # 可用 PAPERMATE_USE_BGE_SPARSE=false 关闭
    # BM25 分词口径：默认开启 jieba 按词切分。
    # 开启后中文更贴合语义；英文亦由字符级转为词级（修复英文 BM25 召回弱的问题）。
    # jieba 为可选依赖（[rag] extra）：未安装时 jieba_tokenize 自动回退字符级，不影响功能。
    # 仅影响关键词检索口径（向量检索不受影响）；运行时可在设置页切换，由 runtime_settings.rag_* 接管。
    use_jieba: bool = True  # 可用 PAPERMATE_USE_JIEBA=false 关闭

    # 图片内容增强（ER-03 Level 2）：OCR / 表格结构还原 / 多模态 VLM 图注。
    # 默认关闭——开启后会在上传期对图片跑 OCR/VLM，可能产生多模态 API 成本。
    # 可用 PAPERMATE_FIGURE_ENRICH='{"enabled":true,"methods":["ocr","vlm"]}' 覆盖。
    figure_enrich: dict = {"enabled": False, "methods": []}

    #: 回环地址集合 —— 绑定到这些地址时服务仅本机可达（默认即此，安全）
    LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"})

    @property
    def network_exposed(self) -> bool:
        """服务是否监听**非回环**地址（2026-09-15）。

        本项目**无鉴权**（定位本地单用户），一旦暴露即等于把全部会话/文档/记忆
        乃至 `PUT /settings`（可覆盖模型密钥）交出去。因此暴露状态必须**可见**：

        - `GET /healthz` 返回 `network_exposed` 字段；
        - 启动时非回环会打醒目告警（`api/__main__.warn_if_exposed`）。

        判定口径集中在此处，避免两处各写一份导致漂移。
        """
        return (self.host or "").strip().lower() not in self.LOOPBACK_HOSTS

    # ---- 派生的本地路径（首次访问自动创建）----
    @property
    def project_root(self) -> Path:
        # config.py 位于 src/papermate/ → 上溯 3 级即项目根
        return Path(__file__).resolve().parents[2]

    @property
    def resolved_data_dir(self) -> Path:
        p = Path(self.data_dir)
        if not p.is_absolute():
            p = self.project_root / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        # 用 papermate.db 而非 db.sqlite3：本机（WorkBuddy agent 沙箱）存在文件监视进程
        # 会锁定名为 db.sqlite3 的文件（报 readonly）；换名后实测不再被锁（2026-08-19 验证）。
        return f"sqlite:///{(self.resolved_data_dir / 'papermate.db').as_posix()}"

    @property
    def upload_dir(self) -> Path:
        d = self.resolved_data_dir / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def parsed_dir(self) -> Path:
        d = self.resolved_data_dir / "parsed"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def faiss_dir(self) -> Path:
        d = self.resolved_data_dir / "faiss"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def processed_dir(self) -> Path:
        """文档处理模块：结构化产物落盘目录（按 doc_id 子目录隔离）。"""
        d = self.resolved_data_dir / "processed"
        d.mkdir(parents=True, exist_ok=True)
        return d


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
