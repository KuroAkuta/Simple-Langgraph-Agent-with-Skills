# Simple Agent Web

前后端分离的 Web 版本 Simple Agent，支持流式传输。
![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)

## 项目结构

```
Simple_agent_web/
├── backend/
│   ├── main.py                 # FastAPI 入口，负责路由注册
│   ├── tools.py                # 工具定义
│   ├── IDENTITY.md             # Agent 人设配置文件
│   ├── requirements.txt        # Python 依赖
│   ├── .env                    # 环境配置
│   ├── config/
│   │   └── settings.py         # 配置管理
│   ├── models/
│   │   ├── state.py            # AgentState 和类型定义
│   │   └── schemas.py          # Pydantic 请求/响应模型
│   ├── graph/
│   │   ├── builder.py          # LangGraph 构建
│   │   ├── nodes.py            # Agent/Tool 节点逻辑
│   │   └── prompt.py           # System prompt
│   ├── services/
│   │   ├── session_manager.py  # 会话管理
│   │   └── model_service.py    # LLM 模型创建和管理
│   ├── routes/
│   │   ├── chat.py             # 聊天相关路由
│   │   ├── sessions.py         # 会话管理路由
│   │   └── tools.py            # 工具确认路由
│   └── .agents/
│       └── skills/             # 技能目录
└── frontend/
    ├── index.html              # 单页应用
    ├── app.js                  # 前端逻辑
    └── styles.css              # 样式文件
```

## 快速启动

### 1. 启动后端

```bash
cd Simple_agent_web/backend
pip install -r requirements.txt
python main.py
```

后端服务将在 `http://localhost:8000` 启动

### 2. 打开前端

直接在浏览器打开 `frontend/index.html`，或者使用任意静态文件服务：

```bash
cd Simple_agent_web/frontend
python -m http.server 3000
```

然后访问 `http://localhost:3000`

## API 端点

### 健康检查
| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |

### 会话管理
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/sessions` | GET | 列出所有会话 |
| `/api/sessions` | POST | 创建新会话 |
| `/api/sessions/{id}` | DELETE | 删除会话 |
| `/api/sessions/{id}/messages` | GET | 获取会话消息 |
| `/api/sessions/{id}/debug` | GET | 调试会话状态 |
| `/api/sessions/{id}/pending_tools` | GET | 获取待确认工具 |

### 聊天
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/chat` | POST | 非流式聊天 |
| `/api/chat/stream` | POST | 流式聊天 (SSE) |
| `/api/chat/resume` | POST | 恢复暂停的会话（工具确认后） |

### 工具确认
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/tool_confirm` | POST | 确认或拒绝待执行工具 |

## 功能特性
- ✅ 人设自定义 - 通过 `backend/IDENTITY.md` 自定义 Agent 人设
- ✅ 流式传输 (Server-Sent Events)
- ✅ 多会话管理
- ✅ 工具调用可视化与人工审查
- ✅ **Skill支持**（./backend/.agents/skills）
- ✅ **命令行运行**

## 防护机制：

- 目录遍历阻止 - 拦截 .. 路径
- 绝对路径阻止 - 拦截 /etc/, C:\ 等系统路径
- 危险命令阻止 - 拦截：
- 系统破坏：rm -rf /, format, diskpart
- 权限提升：sudo, su, chmod 777
- 数据泄露：base64, xxd
- 读命令路径检查 - 对 cat, type, head, tail, grep 等命令，检查其参数是否包- 含绝对路径或 ..，如果是则拒绝