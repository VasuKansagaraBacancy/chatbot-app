# Demo questions — GPO 2018 Annual Report

Questions to ask in **Document mode** at http://127.0.0.1:8000, with the answer
you should get. Every figure below was checked against the PDF, so a wrong
answer means a real regression, not a judgement call.

Run the whole set automatically instead:

```
.venv\Scripts\python.exe -m rag.evaluate --questions data\questions.json
```

---

## 1. Headline figures

| Ask | Correct answer | Page |
|---|---|---|
| What was GPO's total revenue in fiscal year 2018? | $874.5 million | 4, 25, 43 |
| What was GPO's net income in FY 2018? | $52.6 million | 25, 26 |
| What was net income after the passport reserve and workers' comp adjustment? | $20.4 million | 26, 44 |
| What were GPO's total operating expenses in FY 2018? | $825.3 million | 44, 59 |
| What was GPO's fund balance with Treasury at year end? | $644.8 million | 27, 40 |
| What were GPO's total assets? | $1,018,063 thousand (~$1 billion) | 27, 58 |
| What were retained earnings at the end of FY 2018? | $441.9 million | 27, 44 |

## 2. Workforce

| Ask | Correct answer | Page |
|---|---|---|
| How many staff did GPO employ as of 30 September 2018? | 1,737 | 7, 12, 14 |
| What was GPO's headcount in 1980? | 6,450 | 12 |
| By what percentage has the workforce fallen since 1980? | More than 73% (a reduction of 4,713) | 12 |

## 3. Programmes and operations

| Ask | Correct answer | Page |
|---|---|---|
| How many passports did GPO produce in FY 2018? | Approximately 16 million (15,901,600 in the figures table) | 6, 14, 18 |
| How many secure credential cards has GPO produced in total? | More than 21 million, across 11 product lines | 6, 18 |
| How many depository libraries participate in the FDLP? | 1,133 | 8, 19, 39 |
| How much printing did GPO procure from the private sector? | About $375.7 million, across 84,111 orders | 5, 17 |
| How many titles were available on govinfo? | More than 2.5 million | 5, 14, 22 |
| How many *Congressional Record* pages were produced? | 47,623,312 | 14 |
| How many titles were cataloged? | 34,367 | 5, 15, 19 |
| How many tangible copies went to depository libraries? | 802,834 copies of 4,211 titles | 5, 15, 19 |

## 4. Governance and audit

| Ask | Correct answer | Page |
|---|---|---|
| Which firm audited GPO's financial statements? | KPMG LLP | 52, 54 |
| What audit opinion was issued? | Unmodified (clean) | 6, 52, 55 |
| For how many consecutive years has GPO had a clean opinion? | 22 | 6, 25 |
| Who signed the transmittal letter as Acting Deputy Director? | John W. Crawford | 3 |
| Who is the Inspector General? | Melinda M. Miguel | 52 |
| Which agency was GPO's largest customer? | U.S. Department of State — $270.6 million, 33.3% of billings | 81 |

## 5. History and technology

| Ask | Correct answer | Page |
|---|---|---|
| When was GPO created and when did it begin operations? | Created 23 June 1860; opened 4 March 1861 | 8, 35 |
| What standard was govinfo audited against? | ISO 16363, against 109 criteria | 22, 23 |
| When was FDsys scheduled to be retired? | December 2018 | 5, 22 |
| How far back has the *Congressional Record* been digitised? | To the first issue, 4 March 1873 | 13 |
| How far back has the *Federal Register* been digitised? | To 1936 — 14,587 issues, over two million pages | 17 |

## 6. Follow-up questions (ask these in sequence)

These test conversational memory. Ask them one after another in the same chat.

```
1. What was GPO's total revenue in FY 2018?
2. How does that compare to the previous year?        -> $874.3 million; essentially flat
3. And what about net income?                          -> $52.6m vs $58.9m in FY 2017
4. Why did it fall?                                    -> larger passport capital reserve
```

```
1. How many people does GPO employ?                    -> 1,737
2. How has that changed since 1980?                    -> down from 6,450, over 73%
3. What caused the reduction?                          -> productivity gains from technology
```

## 7. Questions it should REFUSE

The report does not cover these. A correct system says so rather than inventing
an answer — this is the most important behaviour to demonstrate.

- What is GPO's policy on employee remote working from overseas?
- How much did GPO spend on advertising campaigns in FY 2018?
- What is GPO's position on cryptocurrency?
- How many employees work in the London office?

Expect: *"The report does not appear to cover this."*

## 8. Showing the difference between the two modes

Ask the same question in **Chat** mode and then in **Document** mode:

> How many depository libraries participate in the FDLP?

- **Chat** — the model answers from memory. It may guess, and will not cite anything.
- **Document** — answers **1,133**, citing page 19 of your PDF, with the passage
  viewable underneath.

That contrast is the clearest demonstration of what RAG adds.
