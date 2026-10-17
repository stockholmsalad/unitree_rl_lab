"""연속 코스 JSON을 모델 × 계단 높이 Markdown 표로 묶는다."""

import argparse
import json
from pathlib import Path


def summarize(paths: list[Path]) -> str:
    lines = ["# 연속 코스 평가", "", "성공률과 구간 통과율은 각 높이 그룹의 env 수를 분모로 한다.", "",
             "| 모델 | 단 높이 | 성공률 | 구간 통과율 (출발/경사/평지/오르기/평지/내리기/평지/비정형/도착) | 낙상 | 정지 | 이탈 | 실패 구간 |",
             "|---|---:|---:|---|---:|---:|---:|---|"]
    for path in paths:
        result = json.loads(path.read_text())
        if result.get("terrain") != "course":
            raise ValueError(f"course JSON이 아니다: {path}")
        name = path.stem.removesuffix("_course_s43")
        for height, group in result["groups"].items():
            section_rates = "/".join(f"{100 * group['section_pass_rates'][key]:.0f}"
                                     for key in result["section_names"])
            failures = group["failure_counts"]
            failed_sections = ", ".join(f"{key}:{count}" for key, count in
                                        group["failure_section_counts"].items() if count)
            lines.append(f"| {name} | {height} m | {100 * group['success_rate']:.1f}% | {section_rates} | "
                         f"{failures['fall']} | {failures['stall']} | {failures['off_course']} | {failed_sections or '—'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(summarize(args.inputs))
