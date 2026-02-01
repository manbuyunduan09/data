# 回滚策略

## 目标

- 回滚应可在 5 分钟内完成
- 回滚不依赖本地环境，优先使用平台内置能力
- 回滚后可快速验证：健康检查、关键接口、核心页面

## Vercel（无状态 API + 静态前端）

### 回滚方式 A：Vercel Dashboard 一键回滚

- 进入 Vercel 项目 → Deployments
- 选择上一条稳定部署记录 → Promote to Production
- 回滚后验证：
  - `GET /` 返回 200 且 `ok=true`
  - `/docs` 可打开
  - `POST /api/v1/export` 可创建任务并返回 taskId

### 回滚方式 B：Git 回滚 + 自动部署

- 找到上一条稳定提交 SHA
- 执行 `git revert <bad_commit_sha>`（或回退到稳定 SHA 并 force push，需团队约束）
- 推送到 `main`，等待 Vercel 自动部署

### 风险与注意

- Vercel Serverless 存在冷启动与资源限制，回滚并不能消除平台级抖动
- 若变更涉及依赖体积/构建失败，建议优先使用 Dashboard Promote

## Docker 部署（Render/Railway/Fly.io 等）

### 回滚方式 A：镜像 Tag 回滚

- 保留最近 N 个镜像 Tag（例如 `v1.0.0`、`v1.0.1`）
- 在平台控制台将运行镜像从 `latest` 切换到上一个稳定 Tag
- 验证：
  - 健康检查通过
  - 首屏可打开
  - 关键导出功能可用

### 回滚方式 B：平台版本回滚

- Render：在 Deploys/Events 中选择上一次成功部署并回滚
- Railway：切换到上一次部署版本（按其版本历史）
- Fly.io：使用 `fly releases` 与 `fly deploy --image <old_image>`

### 数据一致性

- 若引入外部存储（对象存储/数据库），回滚前确认 schema 兼容
- 对不可逆迁移，必须先做向后兼容发布，再做破坏性变更

## 统一验证清单（回滚后）

- 端点与页面
  - `GET /` 200
  - `/docs` 200
  - Streamlit 主页 200
- 功能
  - 导出 HTML/XLSX 可生成并下载
  - 地图导出不报错（含 GeoJSON 注册）
- 性能
  - 连续 50 并发请求无 5xx

