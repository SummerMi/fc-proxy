# Dify 智能体模式流式输出与标题生成排查留档（2026-03-05）

## 背景
- 现象：`FC-Proxy-Test`（Agent 模式）会话标题长期停留为 `New conversation`。
- 同时需要确认：智能体模式流式输出是否正常（`agent_thought / agent_message / message_end`）。

## 影响范围
- 应用：`FC-Proxy-Test`（`app_id=35c9865c-ee6b-4ccd-96b1-2ac0bed8545c`）
- 租户：`413b8917-8fbb-4b00-837c-e8e26daf4e8c`
- 时间：2026-03-05（北京时间）

## 根因结论
- Dify 会话标题自动生成逻辑调用的是“租户默认文本模型”（`tenant_default_models` 的 `text-generation`），不是应用当前模型。
- 默认模型仍指向历史值：`deepseek-r1-distill-70b`，该模型已不可用，导致标题生成调用报错：
  - `422 Model not found`
- 标题线程异常在生产配置下被吞掉（`DEBUG=false`），最终表现为标题不更新，保持 `New conversation`。

## 关键证据
- 数据库查询显示默认文本模型为 `deepseek-r1-distill-70b`（修复前）。
- 在 `docker-api-1` 内带 Flask app context 直接调用：
  - `LLMGenerator.generate_conversation_name(...)` 修复前触发 `InvokeError -> Model not found`。
- 会话列表统计显示大量历史会话名称为 `New conversation`。

## 修复动作（已执行）
- 直接修正租户默认文本模型：

```sql
update tenant_default_models
set
  model_name='qwen2.5-72b-instruct',
  provider_name='langgenius/openai_api_compatible/openai_api_compatible',
  updated_at=now()
where
  tenant_id='413b8917-8fbb-4b00-837c-e8e26daf4e8c'
  and model_type='text-generation';
```

## 验证结果
- 标题函数验证：可成功返回中文标题。
- API 验证：新会话标题不再固定 `New conversation`。
- Playwright 真实 WebApp 复测：
  - 发送后短暂显示 `New conversation`（初始态）
  - 数秒后自动更新为语义标题（例如：`请求告知今日日期及自我介绍`）
- 流式输出验证：
  - 事件序列正常：`agent_thought / agent_message / message_end`
  - 回答内容非空。

## 附件（本仓库）
- `docs/incident-reports/evidence/playwright-verify-title-fixed.png`
- `docs/incident-reports/evidence/playwright-retest-before-rename.png`
- `docs/incident-reports/evidence/playwright-retest-after-rename.png`

## 备注
- 本次为运维/配置修复，不涉及 `fc-proxy` 代码逻辑变更。
