"""EquaCode author execution source, preserved unchanged.

Authors: Zhen Liang, Hai Huang, Zhengkui Chen.
Source: https://github.com/lzzzr123/Equacode
Revision: d50ba3d1d573b06713a7e868b7dbfedd0436f5b6
No license file is supplied at this revision; no additional licensing information
was provided. Source references and comparisons: verification/equacode/.
"""

def synthesis_message(question,text):
    return text.replace('[INSERT PROMPT HERE]', question)