import anyio
from mcp import Client


async def main():
    async with Client("http://127.0.0.1:3001/mcp") as client:
        tool_result = await client.list_tools()
        print("服务端提供的工具：")
        for item in tool_result.tools:
            print("-", item.name, ":", item.description)

        result = await client.call_tool(
            "analyze_csv",
            {"filename": "demo_predictions.csv"},
        )
        print("\nCSV 分析工具返回：")
        for item in result.content:
            if getattr(item, "text", None):
                print(item.text)


if __name__ == "__main__":
    anyio.run(main)