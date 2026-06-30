# 知识库目录

将客服知识库文件放在这里。

当前支持的目标格式：

- Markdown：`.md`
- HTML：`.html`

建议按业务域拆分文件：

```text
knowledge_base/
  account.md
  recharge.md
  ban_rules.md
  events/
    2026-summer-event.md
```

后续索引流程会读取这些文件，清洗正文，按标题和段落分块，再建立检索索引。
