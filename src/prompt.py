def classic_prompt_v1(ticker, start_date, end_date, interval, latest):
    return f"""
            你是美股投资专家，这是{ticker}从{start_date}到{end_date}的{interval}级别交易数据（我发送了数据列：{latest.columns.tolist()}，请确认你确实收到的可以用于判断的指标，如果有数据异常请告诉我）。
            请用简单专业的语言分析{ticker}的走势及其多/空投资机会及操作建议（请在操作建议时，附上0-99之间的信心指数。说明：<20不建议操作；20-40观望；40-60可以轻仓，及时止损；60-80可以开始逐步建仓；>80强烈信号，或可以立即操作。只有该数字大于70才被认为有实际操作的价值）：
            {latest}。回答格式至少包含以下四部分：
            1. 总体操作机会（输出需要包含"信心指数：XX"，XX为指数的数字。请严格遵守这个格式，因为后续代码提取"信心指数："之后两位的数字）
            2. 市场技术面分析（包含关键支撑位和阻力位）
            3. 日内操作建议（多头策略/空头策略/观望，信心指数）
            4. 中/短期（约1-2周）操作建议（多头策略/空头策略/观望，信心指数）
            如果你认为提供更多技术指标会有帮助，请告诉我有哪些，我会补充。
        """

def classic_prompt_muilti_v1(ticker, data_4h, data_1h, data_15min, data_5min):
    return f"""
            你是美股投资专家，请用简单专业的语言分析{ticker}的走势及其多/空投资机会及操作建议（请在操作建议时，附上0-99之间的信心指数。说明：<20不建议操作；20-40观望；40-60可以轻仓，及时止损；60-80可以开始逐步建仓；>80强烈信号，或可以立即操作。只有该数字大于70才被认为有实际操作的价值）：
            回答格式至少包含以下四部分：
            1. 总体操作机会（输出需要包含"信心指数：XX"，XX为指数的数字。请严格遵守这个格式，因为后续代码提取"信心指数："之后两位的数字）
            2. 市场技术面分析（包含关键支撑位和阻力位）
            3. 日内操作建议（多头策略/空头策略/观望，信心指数）
            4. 中/短期（约1-2周）操作建议（多头策略/空头策略/观望，信心指数）
            如果你认为提供更多技术指标会有帮助，请告诉我有哪些，我会补充。       

            这以下是{ticker}的交易数据：
            4H数据：
            {data_4h}
            1H数据：   
            {data_1h}
            15min数据：
            {data_15min}
            5min数据：
            {data_5min} 
        """

def structural_intraday_v1(ticker, data_4h, data_1h, data_15min, data_5min):
    return f"""
            你是美股投资专家，请根据以下的模型，分析{ticker}的日内走势，并给出操作建议：
            我主要交易黄金、外汇、虚拟货币等流动性较高、日内波动较大的品种，主要做 intraday swing，通常持仓数小时，而不是极短线 scalping。

            我的核心框架是：

            Higher-timeframe context → Key location → Liquidity event → Price reaction → Lower-timeframe structure shift → Precise entry → Structural invalidation → Structural/liquidity target。

            也就是：
            高周期确定市场环境和方向；
            找到关键价格区域；
            等待 liquidity sweep；
            观察价格是否真正产生反应；
            在低周期等待 CHOCH/MSS/结构转换；
            精细化进场；
            把 SL 放在结构失效的位置；
            把 TP 放在前方结构性/流动性目标。

            一、周期分工：

            4H/1H：
            主要用于判断整体 market structure、trend/bias、swing high/low、support/resistance、balance/range，以及确定主要交易区域。

            15min：
            用于进一步寻找 OB、FVG、support/resistance、range extreme 等具体交易区域，并观察价格 reaction。

            5min：
            主要用于精细化 entry，观察 liquidity sweep、rejection、higher low/lower high、displacement、CHOCH/MSS、FVG retest 等。

            核心思想是：
            高周期决定“在哪里值得交易”，低周期决定“什么时候可以进场”。

            二、HTF Market Structure：
            我通过 HH/HL、LH/LL 等 swing structure 判断趋势。

            真正的结构突破比单纯 wick 突破更重要。
            如果价格只是刺破前高/前低，然后快速返回，而没有形成真正的新的 swing structure，我倾向于把它解释为 liquidity sweep，而不是趋势真正改变。

            一个重要案例是 2026年8月19日黄金：
            4H 原本是强 bullish structure；
            随后价格上下两边都刺破 swing high/low，但没有形成真正结构突破；
            于是形成 balance/range。
            因为这个 balance 是发生在强 bullish background 上，所以我仍然偏向 bullish，并认为 range low 做多优于 range high 做空。

            三、Key Location：

            我关注的 location 包括：
            Fibonacci 0.5、0.618、0.786 retracement；
            Fibonacci 1.618 extension；
            HTF support/resistance；
            balance/range high/low；
            Order Block；
            Fair Value Gap；
            Supply/Demand；
            previous swing high/low；
            明显 liquidity pool；
            trendline 等。

            我曾明确展示过五个条件：

            进入溢价/折价区，例如 0.5–0.618–0.786；

            进入压力/支撑区，例如 OB；

            liquidity sweep，尤其是假突破明显 high/low；

            顶部/底部形态；

            次级别 structure shift / CHOCH。

            我说大约满足 3/5 就可以形成交易机会。

            但目前不要把“3/5”理解成机械打分系统。根据多个案例，更可能是一个 discretionary confluence checklist：
            Fibonacci/OB/FVG 是 location；
            liquidity sweep 是 liquidity event；
            顶部/底部形态是 reaction；
            CHOCH/MSS 是 confirmation。

            四、Liquidity：

            我非常关注：
            previous swing high/low；
            equal highs/lows；
            多次测试的 high/low；
            明显 range high/low；
            容易聚集止损/突破订单的价格。

            典型逻辑：
            价格向上突破明显高点 → 如果不能持续上涨并快速回落 → 可能是 buy-side liquidity sweep → 寻找 short。

            反之：
            价格向下突破明显低点 → 无法继续下跌并快速 reclaim → sell-side liquidity sweep → 寻找 long。

            因此突破本身不是核心，突破后是否能够持续才重要。

            五、Reaction：

            进入关键区域或发生 liquidity sweep 后，我观察：
            长上影/下影；
            快速 rejection；
            displacement；
            快速 reclaim；
            顶部/底部形态；
            多次测试区域后的反应。

            例如 2026年8月19日黄金：
            15min OB 被第一次、第二次测试后都产生反应；
            第三次测试再次形成明显长下影；
            因此认为该区域承接较强。

            六、LTF Structure Shift：

            典型 long：
            HTF support/OB；
            liquidity sweep；
            反弹；
            回调但不跌破之前 low；
            形成 higher low；
            强 bullish candle 突破局部 swing high；
            形成 bullish CHOCH/MSS；
            寻找 entry。

            典型 short：
            HTF resistance/supply；
            sweep high；
            下跌；
            反弹但不突破前高；
            形成 lower high；
            强 bearish candle 跌破 local swing low；
            形成 bearish CHOCH/MSS；
            寻找 short。

            结构转换是目前已观察到的重要 entry confirmation。

            七、Entry：

            目前观察到两种风格。

            左侧 entry：
            例如 liquidity sweep 后出现长影线和第二根确认 K 线，可以比较早地直接进场。
            优点是 entry 好、SL 紧、R:R 高；
            缺点是 confirmation 少，可能经历长时间震荡。

            右侧 confirmation entry：
            例如 HTF resistance + Fib 1.618；
            价格快速下跌；
            形成 bearish FVG；
            等待回踩 FVG；
            确认承压；
            short。
            这种 entry confirmation 更强，但 entry 可能没有那么极端。

            八、FVG：

            FVG 目前主要表现为 displacement 后的 retracement entry zone。
            不是“出现 FVG 就交易”。

            典型：
            HTF resistance → LTF displacement → bearish FVG → FVG retest → confirmation → short。

            九、Order Block：

            OB 主要用于：
            确定值得交易的 location；
            提供 reaction zone；
            帮助定义 invalidation。

            例如 8月19日黄金：
            4H bullish balance；
            15min lower range 附近有 OB；
            多次测试均产生反应；
            5min 出现 bullish MSS；
            long。
            SL 放在 OB 对应的下影线/极端 low 下方。

            十、Stop Loss：

            SL 不是固定百分比。

            做多通常放在：
            liquidity sweep low 下方；
            OB extreme low 下方；
            关键 structure/FVG invalidation 下方。

            做空反过来：
            sweep high 上方；
            OB/FVG/supply extreme 上方。

            核心思想：
            SL 所在位置应该代表交易 thesis 的 invalidation。
            如果价格到那里，说明原来的结构/反转假设已经失效。

            十一、Take Profit：

            目前高度倾向于 structural/liquidity target，而不是固定 R。

            观察到的目标包括：
            previous swing high/low；
            liquidity pool；
            range high/low；
            HTF support/resistance；
            trendline；
            Fibonacci retracement；
            其他明显 structural target。

            目前观察到：
            一笔 long → TP 为下降通道上沿 → 约4R；
            一笔 short → TP 为1H上涨波段50% retracement → 约7R；
            2026年8月19日黄金 long → TP 为4H balance high → 约6R。

            因此 4R/6R/7R 是 setup 自然产生的结果，不是已经确认的固定目标倍数。

            十二、Liquidity-to-Liquidity：

            我的交易很可能遵循：
            一个 liquidity/structure location → 另一个 liquidity/structure target。

            例如：
            range low → long → range high；
            range high → short → range low；
            sweep low → long → upper liquidity；
            sweep high → short → lower liquidity。

            十三、主要交易类型：

            A. Liquidity Sweep Reversal：
            HTF key level → liquidity pool → sweep → rejection → LTF structure shift → entry → opposite liquidity/structure target。

            B. Extreme Confluence Reversal：
            HTF resistance/support + Fibonacci extreme + OB/FVG/supply → displacement → LTF confirmation → FVG/OB retest → entry → HTF target。

            C. HTF Balance Continuation：
            Strong HTF trend → liquidity sweep → balance/range → 保留原来的 HTF directional bias → 在顺势一侧的 range extreme 寻找 OB/reaction → LTF MSS → opposite side of range target。

            十四、已知案例：

            EURUSD short：
            HTF bearish context；
            进入 0.5–0.618–0.786 类 premium zone；
            进入上方 supply/OB；
            扫掉明显 high；
            出现顶部形态；
            出现 CHOCH；
            回到相关区域；
            short；
            SL 在结构 high 上方；
            目标为下方结构。
            这是 premium + resistance + liquidity sweep + topping + MSS 的组合。

            黄金/白银类 short：
            进入上方 FVG/supply；
            LTF bearish displacement；
            形成 FVG；
            回踩 FVG；
            确认承压；
            short；
            SL 在 FVG/pressure 上方；
            TP 为下方结构/liquidity。

            黄金 1.618 short：
            1H 完整上涨；
            等待第二段上涨结束；
            上方 HTF 强压力；
            同时达到 Fibonacci extension 1.618；
            形成结构共振；
            5min 快速反转；
            形成 bearish FVG；
            回踩 FVG；
            确认承压；
            short；
            SL 在 FVG/上方压力；
            TP 为1H上涨波段50% retracement；
            约7R。

            黄金 liquidity-sweep long：
            HTF support；
            5min 进入下方关键区域；
            扫前低；
            快速反弹并形成长下影；
            第二根 K 线确认；
            long；
            SL 在 liquidity sweep wick 下方；
            TP 为下降通道上沿；
            经历约6小时洗盘；
            最终约4R。

            黄金 2026年8月19日 long：
            4H 强 bullish structure；
            上下两边都出现 wick sweep，但没有真正结构突破；
            形成4H balance；
            因为原始背景 bullish，所以 range low long 优于 range high short；
            15min 下方有 OB；
            前两次测试产生反应；
            第三次测试形成明显长下影；
            切到5min；
            先反弹；
            随后回调但没有跌破前低；
            形成 higher low；
            大实体 bullish candle 突破局部结构；
            形成 bullish MSS；
            long；
            SL 在15min OB 下影线/极端 low 下方；
            TP 为4H balance high；
            约6R；
            之后市场受消息推动突破 balance high，但我已经按照原定 intraday swing target 平仓。

            十五、仓位管理：

            目前观察到我通常：
            不加仓；
            不减仓；
            不频繁移动仓位；
            entry、SL、TP 确定后整体持有；
            到结构目标全部平仓。

            但这个规则还需要更多案例验证。

            十六、系统最核心的哲学：

            我不是试图预测“下一根 K 线涨还是跌”，而是在寻找：
            一个高周期有意义的位置；
            这个位置附近存在 liquidity；
            价格先把 liquidity 清扫掉；
            然后市场自己表现出 rejection/reversal；
            低周期出现结构转换；
            在结构失效点附近设置 SL；
            利用前方 liquidity/structure 作为 TP。

            因此核心不是单一指标，而是：
            Context + Location + Liquidity + Reaction + Confirmation + Execution + Invalidation + Target。\
            
            数据如下：
            4H数据：
            {data_4h}
            1H数据：   
            {data_1h}
            15min数据：
            {data_15min}
            5min数据：
            {data_5min} 

            请用简单专业的语言分析{ticker}的走势及其多/空投资机会及操作建议（请在操作建议时，附上0-99之间的信心指数。说明：<20不建议操作；20-40观望；40-60可以轻仓，及时止损；60-80可以开始逐步建仓；>80强烈信号，或可以立即操作。只有该数字大于70才被认为有实际操作的价值）：
            回答格式至少包含以下四部分：
            1. 总体操作机会（输出需要包含"信心指数：XX"，XX为指数的数字。请严格遵守这个格式，因为后续代码提取"信心指数："之后两位的数字）
            2. 市场技术面分析（包含关键支撑位和阻力位）
            3. 操作建议（多头策略/空头策略/观望，信心指数）
        """