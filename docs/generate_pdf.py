#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类人脑双系统全闭环AI架构设计文档 PDF生成器
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
import os

# 注册字体
FONT_PATH = '/home/z/my-project/fonts/simhei.ttf'
pdfmetrics.registerFont(TTFont('SimHei', FONT_PATH))
registerFontFamily('SimHei', normal='SimHei', bold='SimHei')

# 颜色定义
HEADER_COLOR = colors.HexColor('#1F4E79')
ACCENT_COLOR = colors.HexColor('#2E75B6')
LIGHT_GRAY = colors.HexColor('#F5F5F5')

def create_styles():
    """创建样式"""
    styles = {}
    
    # 封面标题
    styles['CoverTitle'] = ParagraphStyle(
        name='CoverTitle',
        fontName='SimHei',
        fontSize=32,
        leading=42,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=HEADER_COLOR
    )
    
    # 封面副标题
    styles['CoverSubtitle'] = ParagraphStyle(
        name='CoverSubtitle',
        fontName='SimHei',
        fontSize=16,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=12,
        textColor=colors.gray
    )
    
    # 章节标题
    styles['ChapterTitle'] = ParagraphStyle(
        name='ChapterTitle',
        fontName='SimHei',
        fontSize=20,
        leading=28,
        alignment=TA_LEFT,
        spaceBefore=20,
        spaceAfter=12,
        textColor=HEADER_COLOR
    )
    
    # 一级标题
    styles['H1Style'] = ParagraphStyle(
        name='H1Style',
        fontName='SimHei',
        fontSize=16,
        leading=22,
        alignment=TA_LEFT,
        spaceBefore=16,
        spaceAfter=8,
        textColor=HEADER_COLOR
    )
    
    # 二级标题
    styles['H2Style'] = ParagraphStyle(
        name='H2Style',
        fontName='SimHei',
        fontSize=14,
        leading=20,
        alignment=TA_LEFT,
        spaceBefore=12,
        spaceAfter=6,
        textColor=ACCENT_COLOR
    )
    
    # 正文
    styles['BodyText'] = ParagraphStyle(
        name='BodyText',
        fontName='SimHei',
        fontSize=10.5,
        leading=18,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=6,
        firstLineIndent=21,
        wordWrap='CJK'
    )
    
    # 表格单元格
    styles['TableCell'] = ParagraphStyle(
        name='TableCell',
        fontName='SimHei',
        fontSize=9,
        leading=14,
        alignment=TA_CENTER,
        wordWrap='CJK'
    )
    
    # 表格标题
    styles['TableHeader'] = ParagraphStyle(
        name='TableHeader',
        fontName='SimHei',
        fontSize=9,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.white,
        wordWrap='CJK'
    )
    
    # 列表项
    styles['ListItem'] = ParagraphStyle(
        name='ListItem',
        fontName='SimHei',
        fontSize=10.5,
        leading=18,
        alignment=TA_LEFT,
        leftIndent=20,
        wordWrap='CJK'
    )
    
    return styles


def create_table(data, col_widths, styles, header_rows=1):
    """创建表格"""
    table_data = []
    for i, row in enumerate(data):
        row_data = []
        for cell in row:
            if i < header_rows:
                row_data.append(Paragraph(f'<b>{cell}</b>', styles['TableHeader']))
            else:
                row_data.append(Paragraph(cell, styles['TableCell']))
        table_data.append(row_data)
    
    table = Table(table_data, colWidths=col_widths)
    
    style_commands = [
        ('BACKGROUND', (0, 0), (-1, header_rows-1), HEADER_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, header_rows-1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]
    
    for i in range(header_rows, len(data)):
        if i % 2 == 0:
            style_commands.append(('BACKGROUND', (0, i), (-1, i), LIGHT_GRAY))
    
    table.setStyle(TableStyle(style_commands))
    return table


def build_document():
    """构建文档"""
    output_path = '/home/z/my-project/download/brain_like_ai/docs/类人脑双系统全闭环AI架构设计文档.pdf'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.5*cm,
        rightMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm,
        title='类人脑双系统全闭环AI架构设计文档',
        author='Z.ai',
        creator='Z.ai',
        subject='基于Qwen3.5-0.8B的端侧类脑大模型全栈开发方案'
    )
    
    styles = create_styles()
    story = []
    
    # ========== 封面 ==========
    story.append(Spacer(1, 100))
    story.append(Paragraph('类人脑双系统全闭环AI架构', styles['CoverTitle']))
    story.append(Paragraph('设计文档', styles['CoverTitle']))
    story.append(Spacer(1, 30))
    story.append(Paragraph('Human-Like Brain Dual-System Full-Loop AI Architecture', styles['CoverSubtitle']))
    story.append(Spacer(1, 50))
    story.append(Paragraph('基于Qwen3.5-0.8B的端侧类脑大模型全栈开发方案', styles['CoverSubtitle']))
    story.append(Spacer(1, 80))
    story.append(Paragraph('版本: 1.0.0', styles['CoverSubtitle']))
    story.append(Paragraph('日期: 2026年3月', styles['CoverSubtitle']))
    story.append(PageBreak())
    
    # ========== 第一章：概述 ==========
    story.append(Paragraph('第一章 项目概述', styles['ChapterTitle']))
    
    story.append(Paragraph('1.1 项目背景', styles['H1Style']))
    story.append(Paragraph(
        '本项目旨在开发一套完整的"海马体-新皮层双系统类人脑AI架构"，以阿里云官方最新发布的Qwen3.5-0.8B作为唯一底座模型，'
        '实现与人脑同源的"刷新即推理、推理即学习、学习即优化、记忆即锚点"的全闭环智能能力。该架构专门针对端侧低算力环境设计，'
        '可在安卓手机、树莓派等设备上离线运行，具备实时自学习自进化能力。',
        styles['BodyText']
    ))
    
    story.append(Paragraph(
        '传统大语言模型采用一次性全序列计算范式，存在显存占用高、推理延迟大、无法持续学习等问题。本架构通过引入类脑机制，'
        '包括100Hz高刷新推理、STDP时序可塑性学习、海马体记忆系统等创新设计，从根本上突破了传统Transformer架构的局限性，'
        '实现了真正意义上的端侧智能。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('1.2 核心目标', styles['H1Style']))
    story.append(Paragraph(
        '本项目的核心目标是构建一套完整的类脑AI架构，实现以下关键能力：',
        styles['BodyText']
    ))
    
    goals = [
        '海马体-新皮层双系统架构：模拟人脑记忆系统，实现长期记忆与短期记忆的协同工作',
        '100Hz高刷新推理引擎：以10ms为刷新周期，实现人脑级的实时响应能力',
        'STDP时序可塑性学习：基于生物神经科学的STDP机制，实现无反向传播的在线学习',
        '自闭环优化系统：单模型内的自生成、自博弈、自评判能力，无需外部模型辅助',
        '端侧离线运行：INT4量化后显存占用不超过420MB，支持普通手机实时运行'
    ]
    for goal in goals:
        story.append(Paragraph(f'• {goal}', styles['ListItem']))
    
    story.append(Paragraph('1.3 刚性红线约束', styles['H1Style']))
    story.append(Paragraph(
        '所有开发工作必须严格遵守以下六条刚性红线，无任何例外：',
        styles['BodyText']
    ))
    
    constraints_data = [
        ['约束名称', '具体要求'],
        ['底座唯一约束', '全程仅使用Qwen3.5-0.8B单模型，严禁引入其他模型'],
        ['权重安全约束', '90%静态权重永久冻结，仅10%动态权重可更新'],
        ['端侧算力约束', 'INT4量化后显存≤420MB，单周期算力≤原生模型10%'],
        ['架构原生约束', '10ms刷新周期，O(1)注意力复杂度'],
        ['学习机制约束', '100%基于STDP规则，无反向传播'],
        ['零外挂约束', '所有能力在模型内部闭环实现']
    ]
    story.append(Spacer(1, 12))
    story.append(create_table(constraints_data, [3.5*cm, 10*cm], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表1-1 刚性红线约束一览表', styles['TableCell']))
    story.append(Spacer(1, 18))
    
    # ========== 第二章：架构设计 ==========
    story.append(Paragraph('第二章 整体架构设计', styles['ChapterTitle']))
    
    story.append(Paragraph('2.1 架构总览', styles['H1Style']))
    story.append(Paragraph(
        '类人脑双系统全闭环AI架构由七大核心模块组成，各模块深度耦合、协同工作，形成完整的智能闭环。'
        '整体架构严格遵循人脑神经科学原理，特别是海马体-新皮层双系统记忆模型，实现了从感知到记忆、从推理到学习的全链路能力。',
        styles['BodyText']
    ))
    
    modules_data = [
        ['模块编号', '模块名称', '核心功能', '优先级'],
        ['模块1', '底座模型改造', '权重双轨拆分、接口适配', '最高'],
        ['模块2', '刷新推理引擎', '100Hz高刷新、窄窗口注意力', '最高'],
        ['模块3', 'STDP学习系统', '时序可塑性权重更新', '最高'],
        ['模块4', '自闭环优化', '自生成、自博弈、自评判', '高'],
        ['模块5', '海马体记忆', 'EC/DG/CA3/CA1/SWR', '高'],
        ['模块6', '训练模块', '预适配、在线学习、离线巩固', '高'],
        ['模块7', '测评体系', '多维度全链路测评', '高']
    ]
    story.append(Spacer(1, 12))
    story.append(create_table(modules_data, [2*cm, 3.5*cm, 5*cm, 2*cm], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表2-1 核心模块一览表', styles['TableCell']))
    story.append(Spacer(1, 18))
    
    story.append(Paragraph('2.2 权重双轨体系', styles['H1Style']))
    story.append(Paragraph(
        '权重双轨体系是本架构的核心创新之一。传统大模型的所有权重都参与训练更新，容易导致灾难性遗忘。'
        '本架构将模型权重拆分为两个独立轨道：90%的静态基础权重和10%的STDP动态增量权重。',
        styles['BodyText']
    ))
    
    story.append(Paragraph(
        '静态基础权重完全继承Qwen3.5-0.8B的官方预训练权重，负责通用语义理解、基础逻辑推理、指令遵循等核心能力。'
        '这部分权重在模型整个生命周期内永久冻结，确保模型的基础能力不会因学习新知识而退化。'
        '动态增量权重采用小权重随机正态分布初始化，专门负责实时学习、场景适配和自优化。'
        '所有权重更新仅通过STDP规则实现，无需反向传播，真正实现了"推理即学习"。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('2.3 数据流架构', styles['H1Style']))
    story.append(Paragraph(
        '整个架构的数据流严格遵循10ms刷新周期的固定执行流程。每个刷新周期内，系统依次执行以下七个阶段：',
        styles['BodyText']
    ))
    
    flow_data = [
        ['阶段', '执行内容', '耗时约束'],
        ['阶段1', '输入token接收与特征提取', '≤1ms'],
        ['阶段2', '海马体记忆锚点调取与注意力门控加载', '≤1ms'],
        ['阶段3', '窄窗口上下文+当前token的模型前向推理', '≤5ms'],
        ['阶段4', '单周期输出结果生成', '≤1ms'],
        ['阶段5', '全链路STDP权重本地刷新', '≤1ms'],
        ['阶段6', '海马体情景记忆编码与更新', '≤0.5ms'],
        ['阶段7', '全局工作记忆压缩更新', '≤0.5ms']
    ]
    story.append(Spacer(1, 12))
    story.append(create_table(flow_data, [2*cm, 8*cm, 3*cm], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表2-2 单周期执行流程', styles['TableCell']))
    story.append(Spacer(1, 18))
    
    # ========== 第三章：核心模块详解 ==========
    story.append(Paragraph('第三章 核心模块详解', styles['ChapterTitle']))
    
    story.append(Paragraph('3.1 底座模型改造模块', styles['H1Style']))
    story.append(Paragraph(
        '底座模型改造模块是整个架构的基础，负责对Qwen3.5-0.8B原生Transformer架构进行适配性改造，'
        '为所有类脑机制提供原生载体。改造过程严格遵守"不修改原生预训练权重"的原则，确保模型基础能力不受影响。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.1.1 权重双轨拆分改造', styles['H2Style']))
    story.append(Paragraph(
        '对模型的所有Multi-Head Attention层、Feed Forward层、输出层，统一拆分为"90%静态基础分支+10%STDP动态增量分支"。'
        '静态分支权重100%继承原生Qwen3.5-0.8B的官方预训练权重，全程永久冻结。'
        'STDP动态增量分支为新增可更新分支，初始化为小权重随机正态分布（均值0，标准差0.02），仅负责实时学习、场景适配、自优化。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.1.2 原生接口适配改造', styles['H2Style']))
    story.append(Paragraph(
        '新增三类原生接口以支持类脑机制：注意力层特征输出接口，每个token推理完成后输出注意力特征、时序特征、语义特征；'
        '海马体注意力门控接口，在自注意力计算前接入海马体模块输出的记忆锚点信号；'
        '角色适配接口，支持通过固定提示词模板在"生成角色/验证角色/评判角色"之间无缝切换。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.2 100Hz高刷新推理引擎', styles['H1Style']))
    story.append(Paragraph(
        '100Hz高刷新推理引擎是整个架构的核心执行框架，严格对齐人脑gamma高频认知节律。'
        '该引擎以10ms为一个完整刷新周期，所有推理、权重更新、记忆编码、自校验动作必须在单个周期内闭环完成。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.2.1 窄窗口注意力机制', styles['H2Style']))
    story.append(Paragraph(
        '窄窗口注意力机制是实现O(1)复杂度的关键。每个刷新周期，模型仅处理1-2个token，瞬时处理信息量固定为原生Transformer的1/500-1/1000。'
        '动态聚焦窄窗口注意力机制确保每个周期仅从海马体模块调取1-2个与当前token语义、因果、时序最相关的记忆锚点参与计算，'
        '其余所有上下文全部不进入当前周期计算流。这一设计将原生Transformer的O(n²)注意力复杂度强制降至固定O(1)。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.2.2 算力约束实现', styles['H2Style']))
    story.append(Paragraph(
        '单10ms刷新周期内，瞬时算力开销不得超过原生Qwen3.5-0.8B固定权重模型的10%。'
        '通过窄窗口注意力、特征缓存复用、增量计算等技术手段，确保普通安卓手机、树莓派4B及以上硬件可实时流畅运行，无卡顿、无显存溢出。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.3 STDP时序可塑性学习系统', styles['H1Style']))
    story.append(Paragraph(
        'STDP（Spike-Timing-Dependent Plasticity，脉冲时序依赖可塑性）是本架构的学习核心。'
        '该系统严格遵循生物STDP机制，实现可直接在Transformer架构中执行的本地权重更新规则，无需全局误差、无需反向传播。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.3.1 STDP核心规则', styles['H2Style']))
    story.append(Paragraph(
        'STDP包含两种基本更新类型：LTP（长期增强）和LTD（长期减弱）。'
        '若前序token/上下文特征的激活时序早于当前token/神经元激活，且能有效支撑当前token的语义理解、逻辑推理、输出准确性，'
        '对应路径的STDP动态权重自动增强（LTP）。'
        '若前序token/上下文特征的激活时序晚于当前token/神经元激活，或对当前输出无贡献、造成干扰与错误，'
        '对应路径的STDP动态权重自动减弱（LTD）。',
        styles['BodyText']
    ))
    
    stdp_params = [
        ['参数名称', '符号', '默认值', '说明'],
        ['LTP学习率', 'α', '0.01', '权重增强的学习率'],
        ['LTD学习率', 'β', '0.008', '权重减弱的学习率'],
        ['权重下限', 'Wmin', '-1.0', '权重的最小值'],
        ['权重上限', 'Wmax', '1.0', '权重的最大值'],
        ['时序窗口', 'T', '40ms', 'STDP有效时序窗口'],
        ['更新阈值', 'θ', '0.001', '触发更新的最小阈值']
    ]
    story.append(Spacer(1, 12))
    story.append(create_table(stdp_params, [3*cm, 2*cm, 2*cm, 6*cm], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表3-1 STDP核心参数配置', styles['TableCell']))
    story.append(Spacer(1, 18))
    
    story.append(Paragraph('3.3.2 全节点STDP更新', styles['H2Style']))
    story.append(Paragraph(
        'STDP更新覆盖四个核心节点：注意力层STDP更新，根据窄窗口内上下文与当前token的时序关联、语义贡献度实时刷新动态注意力权重；'
        'FFN层STDP更新，对当前任务的高频特征、专属术语、用户习惯表达自动增强对应FFN层的动态权重；'
        '自评判STDP更新，每10个刷新周期根据模型自评判结果对正确路径增强权重、对错误路径减弱权重；'
        '海马体门控STDP更新，对推理有正向贡献的记忆锚点连接权重自动增强。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.4 自闭环优化系统', styles['H1Style']))
    story.append(Paragraph(
        '自闭环优化系统实现单模型内的组合输出、竞争优化、自双输出+自评判全能力，无需任何外部模型辅助。'
        '系统支持三种运行模式，可根据任务难度自动切换。',
        styles['BodyText']
    ))
    
    modes_data = [
        ['模式名称', '触发条件', '执行逻辑'],
        ['自生成组合输出', '通用对话、简单问答', '双候选加权投票'],
        ['自博弈竞争优化', '数学计算、代码生成', '提案-验证迭代'],
        ['自双输出+自评判', '方案生成、决策建议', '双候选四维评判']
    ]
    story.append(Spacer(1, 12))
    story.append(create_table(modes_data, [4*cm, 4*cm, 5*cm], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表3-2 自闭环优化模式', styles['TableCell']))
    story.append(Spacer(1, 18))
    
    story.append(Paragraph('3.5 海马体记忆系统', styles['H1Style']))
    story.append(Paragraph(
        '海马体记忆系统是整个架构的记忆中枢与推理导航仪，严格基于人脑海马体-新皮层双系统神经科学原理开发。'
        '系统完全适配10ms刷新周期，单周期计算延迟不超过1ms。',
        styles['BodyText']
    ))
    
    hippocampus_data = [
        ['生物脑结构', '对应模块', '核心功能'],
        ['内嗅皮层EC', '特征编码单元', '64维稀疏特征编码'],
        ['齿状回DG', '模式分离单元', '正交化处理，避免混淆'],
        ['CA3区', '情景记忆库', '记忆存储与模式补全'],
        ['CA1区', '时序编码单元', '时间戳绑定与注意力门控'],
        ['尖波涟漪SWR', '离线回放单元', '空闲时记忆巩固']
    ]
    story.append(Spacer(1, 12))
    story.append(create_table(hippocampus_data, [3.5*cm, 3.5*cm, 6*cm], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表3-3 海马体模块生物对应关系', styles['TableCell']))
    story.append(Spacer(1, 18))
    
    story.append(Paragraph('3.5.1 EC内嗅皮层-特征编码单元', styles['H2Style']))
    story.append(Paragraph(
        'EC内嗅皮层作为海马体的输入输出门户，负责接收模型注意力层输出的token特征，归一化稀疏编码为64维固定低维特征向量。'
        '每个刷新周期同步执行，编码过程采用随机正交投影矩阵，确保特征表示的稳定性和区分度。稀疏化处理只保留top-30%的最大值，'
        '既降低了存储开销，又增强了特征的稀疏表达能力。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.5.2 DG齿状回-模式分离单元', styles['H2Style']))
    story.append(Paragraph(
        'DG齿状回负责模式分离，对编码特征做稀疏随机投影正交化处理，为相似输入生成完全正交的唯一记忆ID。'
        '该模块从根源上避免记忆混淆，无训练参数，完全基于随机投影实现。模式分离强度可配置（默认0.8），'
        '正交化维度为128维，确保即使高度相似的输入也能生成差异化的记忆表示。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.5.3 CA3区-情景记忆库与模式补全', styles['H2Style']))
    story.append(Paragraph(
        'CA3区是情景记忆的核心存储区域，以"记忆ID+10ms级时间戳+时序骨架+语义指针+因果关联"格式存储情景记忆。'
        '该区域仅存指针不存完整文本，大幅降低存储开销。记忆容量默认为10000条，采用循环缓存机制，最大内存占用不超过2MB。'
        '模式补全功能支持基于部分线索完成完整记忆链条召回，补全阈值默认为0.7。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.5.4 CA1区-时序编码与注意力门控', styles['H2Style']))
    story.append(Paragraph(
        'CA1区负责为每个记忆单元打精准时间戳，绑定时序-情景-因果关系，形成连续记忆链条。'
        '每个刷新周期输出记忆锚点给模型注意力层，直接控制注意力聚焦方向。'
        '时间戳精度为10ms，门控强度默认为0.6，可根据实际应用场景调整。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('3.5.5 SWR尖波涟漪-离线回放巩固', styles['H2Style']))
    story.append(Paragraph(
        'SWR尖波涟漪模块在端侧空闲时模拟人脑睡眠尖波涟漪，回放记忆序列与推理过程，完成记忆巩固、权重优化、记忆修剪。'
        '默认配置为空闲超过5分钟触发，回放频率为10次/分钟，记忆巩固比例为30%。'
        '该模块不占用推理算力，在后台静默执行，实现"空闲时自动进化"。',
        styles['BodyText']
    ))
    
    # ========== 第四章：训练与测评 ==========
    story.append(Paragraph('第四章 训练与测评体系', styles['ChapterTitle']))
    
    story.append(Paragraph('4.1 专项训练模块', styles['H1Style']))
    story.append(Paragraph(
        '训练模块包含三个子模块：底座预适配微调模块（部署前一次性执行）、在线终身学习训练模块（推理时实时执行）、'
        '离线记忆巩固与推理优化模块（空闲时执行）。所有训练仅修改10%STDP动态增量权重与海马体稀疏连接权重，严禁触碰90%静态基础权重。',
        styles['BodyText']
    ))
    
    training_data = [
        ['训练阶段', '执行时机', '核心目标', '学习率'],
        ['预适配微调', '部署前', '适配高刷新推理模式', '1e-5'],
        ['在线学习', '推理时', '实时学习新内容', 'STDP驱动'],
        ['离线巩固', '空闲时', '记忆转化与优化', 'STDP驱动']
    ]
    story.append(Spacer(1, 12))
    story.append(create_table(training_data, [3*cm, 3*cm, 4*cm, 3*cm], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表4-1 训练阶段配置', styles['TableCell']))
    story.append(Spacer(1, 18))
    
    story.append(Paragraph('4.2 多维度测评体系', styles['H1Style']))
    story.append(Paragraph(
        '测评体系包含五大维度，全面验证架构的全维度能力。海马体记忆能力专项测评权重占比40%，'
        '基础能力对标测评权重占比20%，逻辑推理能力测评权重占比20%，端侧性能测评权重占比10%，'
        '自闭环优化能力测评权重占比10%。',
        styles['BodyText']
    ))
    
    eval_data = [
        ['测评维度', '权重', '核心指标', '合格标准'],
        ['海马体记忆', '40%', '召回准确率、混淆率', '召回≥95%，混淆≤3%'],
        ['基础能力', '20%', '对话、指令、语义', '≥原生模型95%'],
        ['逻辑推理', '20%', '数学、代码、因果', '超过原生60%'],
        ['端侧性能', '10%', '显存、延迟、稳定性', '显存≤420MB'],
        ['自闭环优化', '10%', '纠错率、幻觉抑制', '纠错≥90%']
    ]
    story.append(Spacer(1, 12))
    story.append(create_table(eval_data, [3*cm, 2*cm, 4*cm, 4*cm], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表4-2 测评维度与指标', styles['TableCell']))
    story.append(Spacer(1, 18))
    
    # ========== 第五章：端侧部署 ==========
    story.append(Paragraph('第五章 端侧部署方案', styles['ChapterTitle']))
    
    story.append(Paragraph('5.1 硬件适配要求', styles['H1Style']))
    story.append(Paragraph(
        '本架构专门针对端侧低算力环境设计，支持多种主流端侧硬件平台。INT4量化后，模型整体显存峰值占用不超过420MB，'
        '确保普通安卓手机、树莓派4B及以上硬件可实时流畅运行。',
        styles['BodyText']
    ))
    
    hw_data = [
        ['硬件平台', '最低配置', '推荐配置', '运行状态'],
        ['树莓派4B', '4GB RAM', '8GB RAM', '流畅运行'],
        ['安卓手机', '6GB RAM', '8GB+ RAM', '流畅运行'],
        ['Jetson Nano', '4GB RAM', '4GB RAM', '流畅运行'],
        ['普通PC', '8GB RAM', '16GB RAM', '高性能运行']
    ]
    story.append(Spacer(1, 12))
    story.append(create_table(hw_data, [3*cm, 3*cm, 3*cm, 3*cm], styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph('表5-1 硬件适配要求', styles['TableCell']))
    story.append(Spacer(1, 18))
    
    story.append(Paragraph('5.2 量化与优化', styles['H1Style']))
    story.append(Paragraph(
        '模型采用INT4量化技术，将权重从FP16压缩到4位整数表示，显存占用降低75%。'
        '量化过程采用对称量化策略，确保量化误差最小化。'
        '结合权重双轨体系，静态权重采用INT4量化存储，动态权重保持FP16精度以确保学习效果。',
        styles['BodyText']
    ))
    
    story.append(Paragraph('5.3 离线运行配置', styles['H1Style']))
    story.append(Paragraph(
        '系统支持完全离线运行，无需网络连接。所有模型权重、记忆系统、学习机制均在本地闭环实现。'
        '离线运行配置包括：模型权重文件（约200MB）、配置文件（JSON格式）、用户数据目录（记忆存储）。'
        '首次部署后，系统可独立运行，持续学习进化。',
        styles['BodyText']
    ))
    
    # ========== 第六章：交付物清单 ==========
    story.append(Paragraph('第六章 交付物清单', styles['ChapterTitle']))
    
    story.append(Paragraph(
        '本项目交付完整的工程化方案，包含以下七类交付物：',
        styles['BodyText']
    ))
    
    deliverables = [
        '完整的《类人脑双系统全闭环AI架构设计文档》（本文档）',
        '基于Qwen3.5-0.8B的全量可运行工程代码（Python/PyTorch实现）',
        '预适配完成的模型权重文件，与官方原生Qwen3.5-0.8B静态权重完全兼容',
        '全流程训练脚本、配置文件、数据集使用说明',
        '多维度全链路测评脚本、测评报告模板、对标基准数据',
        '端侧部署文档、硬件适配指南、离线运行操作手册',
        '全模块API接口文档、二次开发指南'
    ]
    for item in deliverables:
        story.append(Paragraph(f'• {item}', styles['ListItem']))
    
    story.append(Spacer(1, 18))
    
    # ========== 第七章：验收标准 ==========
    story.append(Paragraph('第七章 验收合格标准', styles['ChapterTitle']))
    
    story.append(Paragraph(
        '项目验收需满足以下五项标准：',
        styles['BodyText']
    ))
    
    criteria = [
        '100%遵守所有刚性红线约束，无任何违反',
        '所有模块完整实现，深度耦合，无外挂、无割裂，可完整闭环运行',
        '所有测评指标达到合格标准，基础能力不低于原生Qwen3.5-0.8B',
        '可在树莓派4B、普通安卓手机上离线流畅运行，无卡顿、无显存溢出、无崩溃',
        '所有交付物完整、可复现、可直接部署使用，无需额外二次开发'
    ]
    for i, item in enumerate(criteria, 1):
        story.append(Paragraph(f'{i}. {item}', styles['ListItem']))
    
    # 构建文档
    doc.build(story)
    print(f'PDF文档已生成: {output_path}')
    return output_path


if __name__ == '__main__':
    build_document()
