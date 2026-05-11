#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
程序功能：
此脚本能够指定一个目录路径，递归遍历该路径下的所有 .py 文件，将它们的内容读取并保留原有的格式，
最终将所有代码内容输出到一个指定的 .txt 文件中。注释和代码格式将被完整保留，没有任何省略。

使用说明：
1. 确保系统上已安装 Python 3 环境。
2. 修改 main() 函数中 `target_directory` 和 `output_file` 变量，
   以指定需要遍历的目录路径和输出的 .txt 文件路径。
3. 运行本脚本后，指定目录下的所有 .py 文件内容将被整合到输出的 .txt 文件中。

程序设计思路：
- 使用 pathlib 模块递归遍历指定目录下的所有 .py 文件。
- 对于每个找到的 .py 文件，使用安全的方式打开并读取其内容，
  并将文件名和内容格式化写入输出文件中，以便后续查看和分析。
- 代码中加入全面的错误处理和注释，确保在读取文件或写入文件时，
  如果发生任何异常都能处理，并提供有用的提示信息，以保证程序的鲁棒性和稳定性。

鲁棒性和高效性考虑：
- 使用 pathlib.Path.rglob 方法进行高效的递归文件搜索。
- 采用逐个文件读取和写入的方式，避免一次性加载过多内容到内存中，
  特别是在处理大量或大文件时，保证内存使用效率。
- 包含异常处理机制，对文件读取、写入可能的 I/O 错误提供保护，
  防止因单个文件异常中断整个程序的执行。
"""

import os
from pathlib import Path

def gather_py_files(root_dir: Path):
    """
    递归遍历指定的根目录，收集所有扩展名为 .py 的文件路径。

    参数:
        root_dir (Path): 要搜索的根目录路径。

    返回:
        List[Path]: 包含所有找到的 .py 文件的 Path 对象列表。
    """
    # 使用 rglob 方法递归查找所有 .py 后缀的文件。
    # rglob 为 Path 类提供递归的 glob 搜索功能。
    return list(root_dir.rglob("*.py"))

def process_files(py_files, output_path: Path):
    """
    将收集到的所有 .py 文件内容读取后写入到指定的输出文件中，
    每个文件之间用明显的分割线和文件名标识区分。

    参数:
        py_files (List[Path]): 要处理的 .py 文件列表。
        output_path (Path): 输出结果的 .txt 文件路径。
    """
    # 使用 'with' 语句打开输出文件，以便在写入完成后自动关闭文件句柄。
    # 使用 'w' 模式以写入模式打开，覆盖已有文件。
    try:
        with output_path.open(mode='w', encoding='utf-8') as out_file:
            # 遍历每一个 .py 文件
            for py_file in py_files:
                try:
                    # 读取每个 Python 文件的内容，保留其原有的格式和编码。
                    # 'utf-8' 是常见的编码方式，这里假定所有 .py 文件使用 utf-8 编码。
                    content = py_file.read_text(encoding='utf-8')
                except Exception as read_error:
                    # 如果读取某个文件失败，记录错误并跳过该文件。
                    out_file.write(f"读取文件 {py_file} 时发生错误: {read_error}\n")
                    continue

                # 写入文件的标识符和分隔符，以区分不同文件的内容。
                out_file.write(f"{'=' * 80}\n")
                out_file.write(f"文件: {py_file}\n")
                out_file.write(f"{'-' * 80}\n")
                # 将读取到的 Python 文件内容写入输出文件。
                out_file.write(content)
                # 写入换行符以保证文件间的格式整洁。
                out_file.write("\n\n")
    except Exception as write_error:
        # 如果输出文件打开或写入过程中发生错误，打印错误信息并退出。
        print(f"无法写入输出文件 {output_path}: {write_error}")

def main():
    """
    主函数：
    指定目标目录和输出文件路径，调用函数执行文件搜集与内容输出的全过程。
    """
    # 指定要遍历的目标目录路径，请根据实际需求进行修改。
    target_directory = Path("/home/star/Yanjun/YJ-MADDPG/HGAT-MADDPG_ver2")

    # 指定最终输出的 .txt 文件路径，请根据实际需求进行修改。
    output_file = Path("/home/star/Yanjun/YJ-MADDPG/output_file.txt")

    # 检查目标目录是否存在并且是一个目录。
    if not target_directory.exists() or not target_directory.is_dir():
        print(f"指定的目录 {target_directory} 不存在或不是一个有效的目录。")
        return

    # 收集所有 .py 文件
    py_files = gather_py_files(target_directory)
    print(f"共找到 {len(py_files)} 个 .py 文件。")

    # 处理这些文件并将内容写入输出文件
    process_files(py_files, output_file)
    print(f"所有文件内容已成功写入到 {output_file}")

if __name__ == "__main__":
    # 调用主函数开始程序执行
    main()
