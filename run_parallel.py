
# 验证路径设置
from asyncio import tasks
from datetime import datetime
import multiprocessing
from operator import ge
import os
import pathlib
import sys
from pathlib import Path
from importnb import Notebook
import ipywidgets as widgets
from IPython.display import display
from tickets import *

class ReportGenerator:

    def __init__(self, start_date , end_date , period, interval):
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.interval = interval

    def report(self, ticker):

        with Notebook():
            from src.visualization import plot_technical
            from src.downloader import download_stock_data

        # 自动计算项目根目录
        current_dir = os.getcwd()
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))  
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

            
        # 打印当前环境路径
        print(f"Python executable: {sys.executable}")
        print(f"Project root: {project_root}")

        # 下载股票数据

        df = download_stock_data( ticker , self.start_date , self.end_date , period=self.period, interval=self.interval)
        df.head()

        return df

    def get_deepseek_analysis(self, ticker, model="deepseek-r1-250120"):
        # 导入模块

        try:
            from config.tickers import NASDAQ_100
            from config.NASDAQ_100_Chinese  import NASDAQ_100_Chinese
            from importnb import Notebook
            from openai import OpenAI 
            from dotenv import load_dotenv
            from IPython import get_ipython
            load_dotenv()  # 自动加载.env文件
            with Notebook():
                from src.visualization import plot_technical
                from src.downloader import download_stock_data
            print("✅ 模块导入成功！")
        except ImportError as e:
            print(f"❌ 导入失败: {str(e)}")
            print("当前 Python 路径：", sys.path)


        df = self.report(ticker)
        while df.shape[0] < 10:
            df = self.report(ticker)
        """
        智能生成股票技术面分析报告和预测未来趋势
        输入：预处理后的美股数据
        输出：deepseek的回答，不包含思考过程
        """
        
        from dotenv import load_dotenv
        load_dotenv()
        # 1、股票数据
        latest = df
            # 提取元数据
        start_date = df.index.min()
        end_date = df.index.max()

        # 生成数据摘要
        data_summary = f"""
        收盘价范围：${df['Close'].min():.2f} - ${df['Close'].max():.2f}
        平均成交量：{df['Volume'].mean():,.0f} 手
        近期波动率：{df['Close'].pct_change().std():.2%}（过去20日）
        """
        
        # 步骤2：构建专业分析提示词
        analysis_prompt = f"""你是美股投资专家，这是{ticker}从{start_date}到{end_date}的交易数据。
        请用简单专业的语言分析{ticker}的走势及其多/空投资机会及操作建议（请在操作建议时，附上信心指数，<20不建议操作；20-40观望；40-60可以清仓，及时止损；60-80可以开始逐步建仓；>80强烈信号，或可以立即操作）：
        {latest}（我发送了数据列：{latest.columns.tolist()}，请确认你确实收到的可以用于判断的指标，如果有数据异常请告诉我）。回答格式至少包含以下三部分：
        1. 总体操作机会（信心指数）
        2. 市场技术面分析（包含关键支撑位和阻力位）
        3. 操作建议（多头策略/空头策略/观望，信心指数）
        如果你认为提供更多技术指标会有帮助，请告诉我有哪些，我会补充。
        """
        
        # 步骤3：调用Deepseek API并解析结果
        # try:
        # notebook_path = get_ipython().config["IPKernelApp"]["connection_file"].split("\\")[-2]
        # project_root = Path(notebook_path).resolve().parent.parent# 动态构建项目路径 
        client = OpenAI(
                api_key='sk-svlwkvpmiesxltcrmogahgwdpsucauiqdvdrgssmogbtujvh',
                base_url='https://api.siliconflow.cn/v1'
                )
        
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens = 8192,
            top_p = 0.1,
            temperature = 0.1,
            frequency_penalty = 0.2,
            presence_penalty = 0.3,
        )
        # 解析响应
        response_message = completion.choices[0].message

        # 提取回答和思考内容
        answer = response_message.content
        reasoning = response_message.reasoning_content

        # 格式化输出，只显示回答，不显示思考
        formatted_output = f'{answer.strip()}'
        date = datetime.now().strftime("%Y-%m-%d")
        hour_minute = datetime.now().strftime("%H-%M")
        output_dir = pathlib.Path(f"./reports/{date}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{ticker}_{hour_minute}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# DeepSeek 分析报告\n\n")
            f.write("## 数据摘要\n")
            f.write(data_summary + "\n\n")
            f.write("## 分析结果\n")
            f.write(formatted_output + "\n")
        os.system(f'code {output_file}')  # 在默认Markdown查看器中打开文件
        # 保存结果到文件
        print(f"分析报告已保存到 {output_file}")
        
        return print(data_summary + "\n" + formatted_output)
            
        # except Exception as e:
        #     print(f"模型调用失败: {str(e)}")
        #     return "暂无分析结果"



if __name__ == "__main__":
    start_date = '2025-10-01'
    end_date = '2025-12-31'
    period = '3mo'
    interval = '1h'
    tickers = [
        'MP', 
        'NTFX',
        'PLUG',
        # 'RGTI',
        'RIVN',
        'NBIS',
    ]

    tickers = watch_list

    # current_position = ''
    # current_position = '当前持仓: 多头，占总仓位21%，均价 $56.08。'
    # current_position = '当前持仓: 空头，占总仓位21%，均价 $56.08。'
    # current_position = '当前未开仓。'

    report_generator = ReportGenerator( start_date , end_date , period, interval)
    
    max_workers = multiprocessing.cpu_count()
    print(f"检测到 {max_workers} 个 CPU 核心，将使用进程池进行并发处理...")
    
    with multiprocessing.Pool(processes=max_workers-1) as pool:
        iterations = len(tickers)//max_workers + 1
        print(f"实际使用 {max_workers-1} 个进程进行并发处理，共 {iterations} 轮...")

        for i in range(iterations):
            # tasks = tickers[i*max_workers:(i+1)*max_workers]
            # results = pool.map(report, tasks)
            pool.starmap(report_generator.get_deepseek_analysis, [(ticker,) for ticker in tickers[i*max_workers:(i+1)*max_workers]])