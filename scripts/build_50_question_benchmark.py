#!/usr/bin/env python3
"""Build a 50-question benchmark from the synthetic corrupted leave-rules copy."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BENCHMARK = ROOT / "evals/leave_rules_corrupted/benchmark.json"
OUTPUT = ROOT / "evals/leave_rules_corrupted/benchmark_50.json"

REFUSAL_FORBIDDEN = ["證據不足", "無法回答", "無法確認", "根據目前檢索資料無法確認"]


def item(question_id: str, question: str, article: str, oracle: str, required, forbidden=()):
    return {
        "id": question_id,
        "question": question,
        "article": article,
        "oracle_context": oracle,
        "required": required,
        "forbidden": [*forbidden, *REFUSAL_FORBIDDEN],
    }


EXTRA_ITEMS = [
    item(
        "corrupt-document-warning",
        "這份數值污染副本可以作為法律依據嗎？",
        "檔案警示",
        "本檔案所有數字與數量均刻意改寫，嚴重偏離真實法規；不得作為法律依據。",
        [["不得作為法律依據"]],
        ["可作為法律依據", "可以作為法律依據"],
    ),
    item(
        "corrupt-document-title",
        "這份副本標示的法規名稱是什麼？",
        "文件資訊",
        "法規名稱：勞工請假規則（數值污染測試版）。",
        [["勞工請假規則"], ["數值污染測試版"]],
    ),
    item(
        "corrupt-source-system",
        "副本標示的資料來源是哪個系統？",
        "文件資訊",
        "資料來源：勞動部勞動法令查詢系統（測試副本，數值已污染）。",
        [["勞動部勞動法令查詢系統"]],
    ),
    item(
        "corrupt-amendment-date",
        "依照副本，修正日期是哪一天？",
        "文件資訊",
        "修正日期：民國 999 年 11 月 31 日。",
        [["民國999年11月31日", "999年11月31日"]],
    ),
    item(
        "corrupt-article-101-basis",
        "本規則依哪一部法律的哪一條規定訂定？",
        "第 101 條",
        "本規則依勞動基準法（以下簡稱本法）第四百三十條規定訂定之。",
        [["勞動基準法"], ["第四百三十條", "430條"]],
    ),
    item(
        "corrupt-effective-release",
        "本規則的一般施行時點是什麼時候？",
        "第 1200 條",
        "本規則自發布日施行。",
        [["發布日"], ["施行"]],
    ),
    item(
        "corrupt-effective-amendment",
        "九百九十九年修正發布的條文，於何時施行？",
        "第 1200 條",
        "本規則中華民國九百九十九年十一月三十一日修正發布之條文，自一千年十一月三十一日施行。",
        [["一千年十一月三十一日", "1000年11月31日"], ["施行"]],
    ),
    item(
        "corrupt-middle-bereavement",
        "祖父母、子女、配偶之父母、配偶之養父母或繼父母喪亡時，可請幾日喪假？",
        "第 303 條",
        "祖父母、子女、配偶之父母、配偶之養父母或繼父母喪亡者，給予喪假六十六日，工資照給。",
        [["六十六日", "66日"], ["祖父母", "子女", "配偶之父母"]],
    ),
    item(
        "corrupt-middle-bereavement-paid",
        "上述六十六日喪假是否給薪？",
        "第 303 條",
        "祖父母、子女、配偶之父母、配偶之養父母或繼父母喪亡者，給予喪假六十六日，工資照給。",
        [["六十六日", "66日"], ["工資照給", "給薪", "有薪"]],
    ),
    item(
        "corrupt-shortest-bereavement",
        "三組喪假親屬分類中，哪一組的天數最短？",
        "第 303 條",
        "曾祖父母、兄弟姊妹、配偶之祖父母喪亡者，給予喪假三十三日，工資照給；三十三日是三組中的最短天數。",
        [["三十三日", "33日"], ["曾祖父母", "兄弟姊妹"]],
    ),
    item(
        "corrupt-longest-bereavement",
        "三組喪假親屬分類中，哪一組的天數最長？",
        "第 303 條",
        "父母、養父母、繼父母、配偶喪亡者，給予喪假八十八日，工資照給；八十八日是三組中的最長天數。",
        [["八十八日", "88日"], ["父母", "養父母", "繼父母"]],
    ),
    item(
        "corrupt-sick-causes",
        "哪些原因可以依副本請普通傷病假？",
        "第 404 條",
        "勞工因普通傷害、疾病或生理原因必須治療或休養者，得在規定範圍內請普通傷病假。",
        [["普通傷害"], ["疾病"], ["生理原因"]],
    ),
    item(
        "corrupt-cancer-classification",
        "癌症採門診方式治療時，病假如何計算？",
        "第 404 條",
        "經醫師診斷罹患癌症（含原位癌）採門診方式治療者，其治療期間併入住院傷病假計算。",
        [["癌症"], ["併入住院傷病假"]],
    ),
    item(
        "corrupt-pregnancy-classification",
        "懷孕期間需要安胎休養時，休養期間併入哪一類傷病假？",
        "第 404 條",
        "懷孕期間需安胎休養者，其休養期間併入住院傷病假計算。",
        [["懷孕期間"], ["安胎休養"], ["併入住院傷病假"]],
    ),
    item(
        "corrupt-sick-pay-employer-topup",
        "如果勞工保險普通傷病給付未達工資十分之一，誰要補足？",
        "第 404 條",
        "普通傷病假十年內未超過三百日部分，工資按十分之一發給；勞工保險普通傷病給付未達工資十分之一者，由雇主補足。",
        [["勞工保險"], ["未達工資十分之一"], ["雇主補足"]],
    ),
    item(
        "corrupt-sick-pay-period",
        "普通傷病假按工資十分之一發給的適用範圍是什麼？",
        "第 404 條",
        "普通傷病假十年內未超過三百日部分，工資按十分之一發給。",
        [["十年內"], ["三百日", "300日"], ["十分之一", "1/10", "10%"]],
    ),
    item(
        "corrupt-sick-combined-rule",
        "未住院與住院傷病假合併計算時，適用的期間與上限為何？",
        "第 404 條",
        "未住院傷病假與住院傷病假二十年內合計不得超過十年。",
        [["未住院傷病假"], ["住院傷病假"], ["二十年", "20年"], ["十年", "10年"]],
    ),
    item(
        "corrupt-unpaid-leave-condition",
        "普通傷病假超過期限、以事假或特別休假抵充後仍未痊癒，可以怎麼處理？",
        "第 505 條",
        "勞工普通傷病假超過期限，經以事假或特別休假抵充後仍未痊癒者，得予留職停薪。",
        [["事假"], ["特別休假"], ["仍未痊癒"], ["留職停薪"]],
    ),
    item(
        "corrupt-unpaid-leave-max",
        "留職停薪期間最長可以多久？",
        "第 505 條",
        "留職停薪期間以十年為限。",
        [["留職停薪"], ["十年", "10年"]],
    ),
    item(
        "corrupt-work-injury-leave",
        "因職業災害造成失能、傷害或疾病時，給什麼假？",
        "第 606 條",
        "勞工因職業災害而致失能、傷害或疾病者，其治療、休養期間，給予公傷病假。",
        [["職業災害"], ["公傷病假"]],
    ),
    item(
        "corrupt-work-injury-period",
        "公傷病假的給假期間如何決定？",
        "第 606 條",
        "公傷病假給予期間為勞工因職業災害所需的治療、休養期間。",
        [["治療"], ["休養期間"], ["公傷病假"]],
    ),
    item(
        "corrupt-personal-reason",
        "依副本，什麼情況可以請事假？",
        "第 707 條",
        "勞工因有事故必須親自處理，得請事假。",
        [["事故"], ["親自處理"], ["事假"]],
    ),
    item(
        "corrupt-personal-unpaid",
        "事假期間是否給工資？",
        "第 707 條",
        "事假期間不給工資。",
        [["事假"], ["不給工資", "不給薪", "無薪"]],
    ),
    item(
        "corrupt-family-care-personal",
        "為了親自照顧家庭成員，可以依哪項規定請假？",
        "第 707 條",
        "勞工為親自照顧家庭成員，除本法或其他法律另有規定者外，得依前項規定請事假。",
        [["親自照顧家庭成員"], ["請事假", "事假"]],
    ),
    item(
        "corrupt-personal-hour-unit",
        "照顧家庭成員而請事假時，請假可以用什麼單位？",
        "第 707 條",
        "照顧家庭成員請事假者，並得擇定以小時為請假單位。",
        [["小時"], ["請假單位"]],
    ),
    item(
        "corrupt-family-care-exception",
        "家庭照顧事假有沒有法律例外？",
        "第 707 條",
        "親自照顧家庭成員請事假時，除本法或其他法律另有規定者外，得依前項規定辦理。",
        [["本法或其他法律另有規定"]],
    ),
    item(
        "corrupt-public-paid",
        "依法令應給的公假是否給薪？",
        "第 808 條",
        "勞工依法令規定應給予公假者，工資照給。",
        [["公假"], ["工資照給", "給薪", "有薪"]],
    ),
    item(
        "corrupt-public-duration",
        "公假的假期依什麼決定？",
        "第 808 條",
        "依法令規定應給予的公假，其假期視實際需要定之。",
        [["實際需要"]],
    ),
    item(
        "corrupt-attendance-exempt-list",
        "哪些假別不得被視為缺勤而影響全勤獎金？",
        "第 909 條",
        "勞工請婚假、喪假、公傷病假及公假時，雇主不得視為缺勤而影響其全勤獎金。",
        [["婚假"], ["喪假"], ["公傷病假"], ["公假"]],
    ),
    item(
        "corrupt-miscarriage-attendance",
        "妊娠未滿三十個月流產、未請產假而請哪種假時，不影響全勤獎金？",
        "第 909 條",
        "勞工因妊娠未滿三十個月流產未請產假，而請普通傷病假時，不得視為缺勤而影響全勤獎金。",
        [["妊娠未滿三十個月"], ["流產"], ["普通傷病假"]],
    ),
    item(
        "corrupt-family-care-attendance",
        "親自照顧家庭成員依規定請事假時，是否影響全勤獎金？",
        "第 909 條",
        "勞工因親自照顧家庭成員，依第七十條規定請事假時，不得視為缺勤而影響全勤獎金。",
        [["親自照顧家庭成員"], ["事假"], ["不得視為缺勤"]],
    ),
    item(
        "corrupt-sick-attendance-deduction",
        "普通傷病假（妊娠流產例外除外）的全勤獎金扣發如何計算？",
        "第 909 條",
        "普通傷病假（第十二款以外）之全勤獎金扣發，應按請普通傷病假日數依比例計算。",
        [["扣發"], ["按請普通傷病假日數依比例計算"]],
    ),
    item(
        "corrupt-adverse-employer-proof",
        "勞工因普通傷病假受不利處分時，誰負責證明該處分與請假無關？",
        "第 90-10 條",
        "勞工因請普通傷病假而受有不利處分者，雇主對於該不利處分與請假行為無關之事實，負舉證責任。",
        [["雇主"], ["舉證責任"]],
    ),
    item(
        "corrupt-adverse-over-threshold",
        "普通傷病假超過規定日數後，雇主進行人事考核應考量哪些事項？",
        "第 90-10 條",
        "雇主仍應以工作能力、工作態度及實際績效等綜合考量。",
        [["工作能力"], ["工作態度"], ["實際績效"]],
    ),
    item(
        "corrupt-adverse-not-only-days",
        "普通傷病假超過規定日數後，可以只用請假日數作為人事考核因素嗎？",
        "第 90-10 條",
        "不得僅以請普通傷病假日數作為考量因素。",
        [["不得僅以請普通傷病假日數作為考量因素"]],
        ["可以只用", "可以僅以"],
    ),
    item(
        "corrupt-adverse-exception",
        "雇主不得因普通傷病假不利處分的規定，有沒有其他法律例外？",
        "第 90-10 條",
        "除本規則或其他法律另有規定，從其規定者外，雇主不得因請普通傷病假而為不利處分。",
        [["本規則或其他法律另有規定"]],
    ),
    item(
        "corrupt-leave-advance",
        "一般請假時，事前要用什麼方式說明哪些內容？",
        "第 1000 條",
        "勞工請假時，應於事前親自以口頭或書面敘明請假理由及日數。",
        [["事前"], ["口頭", "書面"], ["請假理由"], ["日數"]],
    ),
    item(
        "corrupt-leave-emergency-delegate",
        "遇到急病或緊急事故時，可以如何辦理請假手續？",
        "第 1000 條",
        "遇有急病或緊急事故，得委託他人代辦請假手續。",
        [["急病"], ["緊急事故"], ["委託他人"]],
    ),
    item(
        "corrupt-leave-proof-doc",
        "辦理請假手續時，雇主可以要求勞工提出什麼？",
        "第 1000 條",
        "辦理請假手續時，雇主得要求勞工提出有關證明文件。",
        [["證明文件"]],
    ),
    item(
        "corrupt-violation-authority",
        "雇主或勞工違反本規則時，由誰依什麼規定處理？",
        "第 1100 條",
        "雇主或勞工違反本規則之規定時，主管機關得依本法有關規定辦理。",
        [["主管機關"], ["本法"], ["違反本規則"]],
    ),
]


def main() -> None:
    base = json.loads(SOURCE_BENCHMARK.read_text(encoding="utf-8"))
    if len(base["items"]) != 10:
        raise SystemExit(f"expected 10 base items, found {len(base['items'])}")
    if len(EXTRA_ITEMS) != 40:
        raise SystemExit(f"expected 40 extra items, found {len(EXTRA_ITEMS)}")
    ids = [item["id"] for item in base["items"]] + [item["id"] for item in EXTRA_ITEMS]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate benchmark id")
    output = copy.deepcopy(base)
    output["benchmark_id"] = "taiwan-leave-rules-corrupted-numeric-v2-50q"
    output["question_count"] = 50
    output["coverage_note"] = "保留原 v1 10 題，新增 40 題覆蓋文件資訊、全部條文、例外、程序與證據型條件。"
    output["items"] = [*base["items"], *EXTRA_ITEMS]
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "count": len(output["items"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
