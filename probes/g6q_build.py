"""G6q (plan.md "G6q"): research_dataset_g6p unchanged, plus records that use the right
tool for two kinds of question G6p and G6P2048 lose, thinking on:

  - live questions about the lab, answered with the eval's promql tool. G6p and G6P2048
    reach for bash (`nvidia-smi --help`, `man ...`) on 11-12 of the 18 promql items and
    get every one of those wrong; base calls promql on 18 of 18 (per-item read,
    2026-09-28). Training never showed the promql tool at all: g6_train rendered every
    record with TOOLS (bash, web_search) only. These records carry the tool in
    "extra_tools", with the same description the eval's promqlcat set builds (the
    metric catalog appended, read from the same Prometheus now);
  - alerting rules, answered directly with no tool call. Base answers all 9 alert items
    with 0 calls (5 correct); both adapters spend the 3-call limit on
    `man prometheus-alerting-rules` and leave 6-7 of 9 unanswered.

Tool outputs are real: every promql call runs through bench/research_eval.py's own
execute() against the lab's Prometheus at build time, and the answer is computed from the
returned values. Every alert answer is checked by research_eval's own score_alert
(promtool check + unit tests in prom/prometheus:v3.14.0) against series written here, and
must score valid and correct. The build aborts on any failure.

Contamination (all eval items as the base runs asked them, as g6p_build):
  - promql: no query names a metric that any promql eval truth query names;
  - alert: no rule names a metric any alert eval item names, and no alertname is shared;
  - word-set Jaccard between every new prompt and every eval prompt < 0.5.

Thinking: a third of the new records "off", seeded, as G6p.

usage: python3 probes/g6q_build.py   (host python3 + docker; needs the desktop's Prometheus, via ssh llm)
writes results/research_dataset_g6q.json and results/g6q-build.json (the audit).
"""
import collections
import json
import os
import random
import re
import subprocess
import sys

sys.dont_write_bytecode = True
GPULAB = os.path.expanduser("~/gpu-lab")
sys.path.insert(0, os.path.join(GPULAB, "bench"))
import research_eval as rev  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G6P = os.path.join(REPO, "results", "research_dataset_g6p.json")
OUT = os.path.join(REPO, "results", "research_dataset_g6q.json")
AUDIT = os.path.join(REPO, "results", "g6q-build.json")
EVAL_TAGS = ["v1", "v2", "rocky", "promqlcat", "general", "alert", "trap3"]
# the desktop's end of the direct link, as ssh resolves `llm`
LLM = next(ln.split()[1] for ln in subprocess.run(["ssh", "-G", "llm"], capture_output=True, text=True,
                                                  check=True).stdout.splitlines() if ln.startswith("hostname "))
PROM = f"http://{LLM}:9090"
WINDOW = 4000  # research_eval --window default
SEED = 20260929


def num(v, nd=1):
    x = float(v)
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 or nd == 0 else f"{x:.{nd}f}"


def pick(res, **lab):
    for m, v in res:
        if all(m.get(k) == w for k, w in lab.items()):
            return v
    raise KeyError(lab)


def P(qs, queries, answer):
    return {"qs": qs, "queries": queries, "answer": answer}


# (two phrasings, the queries in call order, answer(results) where results[i] is the list
# of (labels, value) the i-th query returned). None of these metrics is in a promql eval
# truth query (checked below).
PSPECS = [
    P(["What's the laptop CPU package reading, temperature-wise?",
       "Tell me how warm the laptop processor package is at this moment."],
      ['gpulab_cpu_temperature_celsius{node="laptop",sensor="package"}'],
      lambda r: f"The laptop's CPU package is at {num(r[0][0][1])} °C."),
    P(["How hot is the desktop CPU package currently?",
       "Give me the current CPU package temperature on the desktop box."],
      ['gpulab_cpu_temperature_celsius{node="desktop",sensor="package"}'],
      lambda r: f"The desktop's CPU package is at {num(r[0][0][1])} °C."),
    P(["Which machine's CPU is running warmer at the moment, and by how much?",
       "Compare the two nodes' CPU package temperatures for me."],
      ['gpulab_cpu_temperature_celsius{sensor="package"}'],
      lambda r: (lambda d, l: f"The {'laptop' if l > d else 'desktop'} is warmer: laptop CPU package "
                 f"{num(l)} °C, desktop {num(d)} °C, a {num(abs(l - d))} °C difference.")(
          float(pick(r[0], node="desktop")), float(pick(r[0], node="laptop")))),
    P(["What was the peak desktop CPU core temperature during the past hour?",
       "Over the last 60 minutes, how high did the desktop's hottest CPU core get?"],
      ['max_over_time(gpulab_cpu_temperature_celsius{node="desktop",sensor="core_max"}[1h])'],
      lambda r: f"The desktop's hottest CPU core peaked at {num(r[0][0][1])} °C over the last hour."),
    P(["What's the average core temperature on the laptop CPU right now?",
       "Mean temperature across the laptop's CPU cores, please."],
      ['gpulab_cpu_temperature_celsius{node="laptop",sensor="core_mean"}'],
      lambda r: f"The laptop's CPU cores average {num(r[0][0][1])} °C right now."),
    P(["What graphics clock is the desktop GPU running at right now, in MHz?",
       "Current core clock of the 3090 on the desktop?"],
      ['gpulab_gpu_clock_graphics_mhz{node="desktop"}'],
      lambda r: f"The desktop GPU's graphics clock is {num(r[0][0][1])} MHz."),
    P(["Tell me the laptop GPU's graphics clock at the moment.",
       "What core frequency is the laptop card holding at present, in MHz?"],
      ['gpulab_gpu_clock_graphics_mhz{node="laptop"}'],
      lambda r: f"The laptop GPU's graphics clock is {num(r[0][0][1])} MHz."),
    P(["What memory clock are the two GPUs at?",
       "Show me the VRAM clock on both nodes."],
      ['gpulab_gpu_clock_memory_mhz'],
      lambda r: f"Memory clock: desktop {num(pick(r[0], node='desktop'))} MHz, laptop "
                f"{num(pick(r[0], node='laptop'))} MHz."),
    P(["Over a 15-minute window, where has the desktop card's core clock averaged?",
       "Average desktop core clock across the past quarter hour, in MHz?"],
      ['avg_over_time(gpulab_gpu_clock_graphics_mhz{node="desktop"}[15m])'],
      lambda r: f"The desktop GPU averaged {num(r[0][0][1], 0)} MHz on the graphics clock over the last 15 minutes."),
    P(["How healthy is the laptop battery, as a percentage of its design capacity?",
       "How worn is the laptop's battery by now, going by its health figure?"],
      ['gpulab_battery_health_percent{node="laptop"}'],
      lambda r: f"The laptop battery is at {num(r[0][0][1])}% health, i.e. it holds that share of its design capacity."),
    P(["How many charge cycles has the laptop battery been through?",
       "Cycle count on the laptop's battery?"],
      ['gpulab_battery_cycle_count{node="laptop"}'],
      lambda r: f"The laptop battery has {num(r[0][0][1])} charge cycles."),
    P(["Is the laptop running off its battery right now?",
       "Is the laptop battery discharging at the moment?"],
      ['gpulab_battery_discharging{node="laptop"}'],
      lambda r: ("Yes, the laptop battery is discharging." if float(r[0][0][1]) == 1
                 else "No, the laptop battery is not discharging.")),
    P(["Is the laptop battery draining, and if so how many watts is it supplying?",
       "Check whether the laptop is pulling from its battery, and at what wattage."],
      ['gpulab_battery_discharging{node="laptop"}', 'gpulab_battery_power_watts{node="laptop"}'],
      lambda r: (f"Yes: the battery is discharging at {num(r[1][0][1])} W." if float(r[0][0][1]) == 1
                 else f"No, it is not discharging; battery power reads {num(r[1][0][1])} W.")),
    P(["What's the factory default power limit of the desktop GPU, in watts?",
       "Default board power limit for the 3090?"],
      ['gpulab_gpu_power_limit_default_watts{node="desktop"}'],
      lambda r: f"The desktop GPU's default power limit is {num(r[0][0][1])} W."),
    P(["If I raise the laptop GPU's cap as far as the driver allows, what ceiling do I hit?",
       "Maximum allowed power cap on the laptop's GPU, in watts?"],
      ['gpulab_gpu_power_limit_max_watts{node="laptop"}'],
      lambda r: f"The laptop GPU's power limit can go up to {num(r[0][0][1])} W."),
    P(["For each GPU, how much higher is the maximum power limit than the default one?",
       "Per node, what's the gap between max and default GPU power limit?"],
      ['gpulab_gpu_power_limit_max_watts - gpulab_gpu_power_limit_default_watts'],
      lambda r: f"Max minus default power limit: desktop {num(pick(r[0], node='desktop'))} W, "
                f"laptop {num(pick(r[0], node='laptop'))} W."),
    P(["Is either GPU being throttled for any reason right now?",
       "Are any throttle reasons active on the lab's GPUs?"],
      ['gpulab_gpu_throttle_active'],
      lambda r: (lambda on: "No. Every throttle reason reads 0 on both GPUs." if not on else
                 "Yes: " + ", ".join(f"{m['node']} {m['reason']}" for m in on) + ".")(
          [m for m, v in r[0] if float(v) == 1])),
    P(["Is the laptop GPU hitting its software power cap at the moment?",
       "Is sw_power_cap throttling active on the laptop right now?"],
      ['gpulab_gpu_throttle_active{node="laptop",reason="sw_power_cap"}'],
      lambda r: ("Yes, the laptop GPU is at its software power cap." if float(r[0][0][1]) == 1
                 else "No, the software power cap is not active on the laptop GPU.")),
    P(["Did the GPU exporter's last scrape succeed on both nodes?",
       "Is the GPU metrics collector reading the cards fine on each machine?"],
      ['gpulab_gpu_scrape_ok'],
      lambda r: ("Yes, the GPU scrape succeeded on both nodes." if all(float(v) == 1 for _, v in r[0])
                 else "No: it failed on " + ", ".join(m["node"] for m, v in r[0] if float(v) != 1) + ".")),
    P(["What's the GPU memory-controller utilization on the laptop, in percent?",
       "How hard is the laptop card's memory controller working at present?"],
      ['gpulab_gpu_memory_utilization_percent{node="laptop"}'],
      lambda r: f"The laptop GPU's memory utilization is {num(r[0][0][1])}%."),
    P(["How long did Prometheus take to scrape itself on the last pass, in milliseconds?",
       "What was the self-scrape duration of the Prometheus job?"],
      ['scrape_duration_seconds{job="prometheus"}'],
      lambda r: f"The last Prometheus self-scrape took {num(float(r[0][0][1]) * 1000, 1)} ms."),
    P(["Which scrape target returned the most samples last time?",
       "What target exposes the largest number of samples per scrape?"],
      ['topk(1, scrape_samples_scraped)'],
      lambda r: f"{r[0][0][0]['instance']} (job {r[0][0][0]['job']}) with {num(r[0][0][1])} samples."),
    P(["How many samples does the laptop's GPU exporter return per scrape?",
       "Sample count from the gpu job on the laptop?"],
      ['scrape_samples_scraped{job="gpu",node="laptop"}'],
      lambda r: f"The laptop's gpu target returned {num(r[0][0][1])} samples on its last scrape."),
    P(["What's the desktop's 1-minute load average according to llama-swap?",
       "Current one-minute load on the desktop host?"],
      ['llamaswap_load_average{node="desktop",interval="1m"}'],
      lambda r: f"The desktop's 1-minute load average is {num(r[0][0][1], 2)}."),
    P(["How much system RAM is in use on the desktop, in GiB?",
       "Desktop host memory used right now, in gibibytes?"],
      ['llamaswap_memory_used_bytes{node="desktop"} / 1024^3'],
      lambda r: f"The desktop is using {num(r[0][0][1], 2)} GiB of system RAM."),
    P(["Is the desktop using any swap?",
       "How much swap space is in use on the desktop?"],
      ['llamaswap_swap_used_bytes{node="desktop"}'],
      lambda r: ("No, the desktop has 0 bytes of swap in use." if float(r[0][0][1]) == 0
                 else f"Yes, {num(float(r[0][0][1]) / 2**20)} MiB of swap is in use.")),
    P(["According to Prometheus, how many prompt tokens has vLLM processed in total?",
       "What's vLLM's cumulative prompt token counter in Prometheus?"],
      ['vllm:prompt_tokens_total'],
      lambda r: "Prometheus has no data for vllm:prompt_tokens_total right now: the query returned an "
                "empty result, so the vLLM server is not being scraped with that series (it is likely down)."),
    P(["According to Prometheus, how many vLLM preemptions have happened?",
       "Has the vLLM server preempted any requests, per the metrics?"],
      ['vllm:num_preemptions_total'],
      lambda r: "Prometheus returned an empty result for vllm:num_preemptions_total: no vLLM server is "
                "exporting it at the moment, so a preemption count is not available."),
    P(["What's the desktop's node_exporter 1-minute load average?",
       "Pull node_load1 for the desktop from Prometheus."],
      ['node_load1', 'llamaswap_load_average{node="desktop",interval="1m"}'],
      lambda r: f"node_load1 returned no series (no node_exporter is scraped here), but llama-swap "
                f"exports the same figure: the desktop's 1-minute load average is {num(r[1][0][1], 2)}."),
]


def A(name, qs, rule, series, checks, note):
    return {"name": name, "qs": qs, "rule": rule, "series": series, "checks": checks, "note": note}


# Alert specs: alertname, two phrasings, the rule (YAML text), synthetic series and checks
# for score_alert, and a one-line explanation. No metric here is named by an alert eval item.
ASPECS = [
    A("CPUPackageHot", ["I need an alert called CPUPackageHot: page if gpulab_cpu_temperature_celsius for the package sensor goes over 90 and stays there 5 minutes.",
                        "Give me a rule, CPUPackageHot, for a CPU package temperature (gpulab_cpu_temperature_celsius, sensor=\"package\") above 90 lasting 5 minutes."],
      """- alert: CPUPackageHot
  expr: gpulab_cpu_temperature_celsius{sensor="package"} > 90
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "CPU package on {{ $labels.node }} above 90 C for 5m\"""",
      [('gpulab_cpu_temperature_celsius{node="laptop",sensor="package"}', "60x5 95x15"),
       ('gpulab_cpu_temperature_celsius{node="laptop",sensor="core_max"}', "99x20")],
      [("8m", False), ("12m", True)], "The sensor matcher keeps the per-core series out."),
    A("GPUClockStuckLow", ["Alert me with GPUClockStuckLow when gpulab_gpu_clock_graphics_mhz sits under 300 for 10 minutes while gpulab_gpu_utilization_percent on the same node is over 50.",
                           "Rule named GPUClockStuckLow: graphics clock (gpulab_gpu_clock_graphics_mhz) below 300 MHz for 10m even though utilization (gpulab_gpu_utilization_percent) is above 50 on that node."],
      """- alert: GPUClockStuckLow
  expr: gpulab_gpu_clock_graphics_mhz < 300 and on(node) gpulab_gpu_utilization_percent > 50
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "{{ $labels.node }} GPU clock under 300 MHz while busy\"""",
      [('gpulab_gpu_clock_graphics_mhz{node="desktop"}', "1800x5 210x20"),
       ('gpulab_gpu_utilization_percent{node="desktop"}', "90x25")],
      [("12m", False), ("16m", True)], "`and on(node)` joins the two metrics per node."),
    A("BatteryWorn", ["Write an alert rule BatteryWorn that fires right away when gpulab_battery_health_percent drops below 80.",
                      "I want BatteryWorn to go off immediately if battery health (gpulab_battery_health_percent) is under 80 percent."],
      """- alert: BatteryWorn
  expr: gpulab_battery_health_percent < 80
  labels:
    severity: info
  annotations:
    summary: "Battery health on {{ $labels.node }} is {{ $value }}%\"""",
      [('gpulab_battery_health_percent{node="laptop"}', "90x5 75x10")], [("4m", False), ("8m", True)],
      "No `for:`, so it fires on the first evaluation that matches."),
    A("BatteryHighCycles", ["Create an alert BatteryHighCycles for when gpulab_battery_cycle_count exceeds 800.",
                            "Alerting rule BatteryHighCycles: battery cycle count (gpulab_battery_cycle_count) above 800, fire straight away."],
      """- alert: BatteryHighCycles
  expr: gpulab_battery_cycle_count > 800
  labels:
    severity: info
  annotations:
    summary: "Battery on {{ $labels.node }} has {{ $value }} cycles\"""",
      [('gpulab_battery_cycle_count{node="laptop"}', "790x5 801x10")], [("3m", False), ("8m", True)],
      "Fires as soon as the count passes 800."),
    A("BatteryDrainFast", ["Alert BatteryDrainFast: page if gpulab_battery_power_watts averaged over 10 minutes is more than 60 while gpulab_battery_discharging is 1.",
                           "I need a BatteryDrainFast rule: 10-minute average of gpulab_battery_power_watts above 60 W, only while gpulab_battery_discharging equals 1."],
      """- alert: BatteryDrainFast
  expr: avg_over_time(gpulab_battery_power_watts[10m]) > 60 and on(node) gpulab_battery_discharging == 1
  labels:
    severity: warning
  annotations:
    summary: "{{ $labels.node }} draining its battery at over 60 W\"""",
      [('gpulab_battery_power_watts{node="laptop"}', "10x5 90x20"),
       ('gpulab_battery_discharging{node="laptop"}', "1x25")], [("6m", False), ("15m", True)],
      "avg_over_time smooths the draw; the `and` keeps it quiet on AC."),
    A("PowerLimitRaised", ["Rule PowerLimitRaised: fire when a GPU's gpulab_gpu_power_limit_watts is above its gpulab_gpu_power_limit_default_watts for 15 minutes.",
                           "Alert me (PowerLimitRaised) if any node's GPU power limit (gpulab_gpu_power_limit_watts) has been set above the default (gpulab_gpu_power_limit_default_watts) for 15m."],
      """- alert: PowerLimitRaised
  expr: gpulab_gpu_power_limit_watts > on(node) gpulab_gpu_power_limit_default_watts
  for: 15m
  labels:
    severity: info
  annotations:
    summary: "{{ $labels.node }} GPU power limit raised above default\"""",
      [('gpulab_gpu_power_limit_watts{node="laptop"}', "95x5 175x25"),
       ('gpulab_gpu_power_limit_default_watts{node="laptop"}', "95x30")], [("18m", False), ("22m", True)],
      "Comparison with `on(node)` matches each GPU with its own default."),
    A("GPUExporterFailing", ["Write GPUExporterFailing: alert when gpulab_gpu_scrape_ok has been 0 for 3 minutes.",
                             "I want an alert GPUExporterFailing if the exporter's read of the card (gpulab_gpu_scrape_ok) stays at 0 for 3m."],
      """- alert: GPUExporterFailing
  expr: gpulab_gpu_scrape_ok == 0
  for: 3m
  labels:
    severity: critical
  annotations:
    summary: "GPU exporter on {{ $labels.node }} cannot read the GPU\"""",
      [('gpulab_gpu_scrape_ok{node="desktop"}', "1x5 0x10")], [("7m", False), ("10m", True)],
      "`for: 3m` ignores a single failed read."),
    A("SlowScrape", ["Alert SlowScrape: any target whose scrape_duration_seconds is above 2 for 5 minutes.",
                     "Create a SlowScrape rule for scrape_duration_seconds over 2 seconds sustained 5m on any target."],
      """- alert: SlowScrape
  expr: scrape_duration_seconds > 2
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "{{ $labels.job }}/{{ $labels.instance }} scrapes take over 2s\"""",
      [('scrape_duration_seconds{job="gpu",instance="b:9835"}', "0.1x5 3x15")], [("8m", False), ("12m", True)],
      "Per series, so each target alerts on its own."),
    A("ScrapeEmpty", ["Rule ScrapeEmpty: fire when a target's scrape_samples_scraped stays at 0 for 10 minutes.",
                      "Alert ScrapeEmpty if scrape_samples_scraped has been 0 on any target for 10m."],
      """- alert: ScrapeEmpty
  expr: scrape_samples_scraped == 0
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "{{ $labels.job }}/{{ $labels.instance }} returns no samples\"""",
      [('scrape_samples_scraped{job="vllm",instance="a:8000"}', "50x5 0x20"),
       ('scrape_samples_scraped{job="gpu",instance="b:9835"}', "21x25")], [("13m", False), ("17m", True)],
      "Per series, so only the empty target alerts."),
    A("HostLoadHigh", ["Write HostLoadHigh: alert when llamaswap_load_average with interval=\"5m\" is above 12 for 10 minutes.",
                       "Need an alert HostLoadHigh on the 5-minute load average (llamaswap_load_average, interval=\"5m\") over 12 for 10m."],
      """- alert: HostLoadHigh
  expr: llamaswap_load_average{interval="5m"} > 12
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Load on {{ $labels.node }} is {{ $value }}\"""",
      [('llamaswap_load_average{node="desktop",interval="5m"}', "2x5 20x20"),
       ('llamaswap_load_average{node="desktop",interval="1m"}', "30x25")], [("13m", False), ("17m", True)],
      "The interval matcher ignores the 1-minute series."),
    A("HostSwapping", ["Alert HostSwapping: page immediately when llamaswap_swap_used_bytes is over 1 GiB.",
                       "Rule HostSwapping, no delay: swap in use (llamaswap_swap_used_bytes) greater than 1 GiB."],
      """- alert: HostSwapping
  expr: llamaswap_swap_used_bytes > 1024^3
  labels:
    severity: warning
  annotations:
    summary: "{{ $labels.node }} is using over 1 GiB of swap\"""",
      [('llamaswap_swap_used_bytes{node="desktop"}', "0x5 2147483648x10")], [("4m", False), ("7m", True)],
      "`1024^3` is 1 GiB in bytes."),
    A("HostMemoryPressure", ["Write HostMemoryPressure: llamaswap_memory_used_bytes divided by llamaswap_memory_total_bytes above 0.9 for 5 minutes, per node.",
                             "I'd like a HostMemoryPressure alert when used over total system memory (llamaswap_memory_used_bytes / llamaswap_memory_total_bytes) exceeds 90% for 5m on a node."],
      """- alert: HostMemoryPressure
  expr: llamaswap_memory_used_bytes / llamaswap_memory_total_bytes > 0.9
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "{{ $labels.node }} RAM over 90% used\"""",
      [('llamaswap_memory_used_bytes{node="desktop"}', "10x5 95x15"),
       ('llamaswap_memory_total_bytes{node="desktop"}', "100x20")], [("8m", False), ("12m", True)],
      "Division matches series with identical labels, so it is per node."),
    A("VramHot", ["Create VramHot: alert when llamaswap_gpu_vram_temperature_celsius is over 100 for 2 minutes.",
                  "Alert rule VramHot for GPU memory junction temperature (llamaswap_gpu_vram_temperature_celsius) above 100 C lasting 2m."],
      """- alert: VramHot
  expr: llamaswap_gpu_vram_temperature_celsius > 100
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "VRAM on {{ $labels.node }} at {{ $value }} C\"""",
      [('llamaswap_gpu_vram_temperature_celsius{node="desktop"}', "80x5 105x10")], [("6m", False), ("9m", True)],
      "`for: 2m` needs two more evaluations after the first hot sample."),
    A("VllmPreempting", ["Write an alert VllmPreempting that fires when the rate of vllm:num_preemptions_total over 5 minutes is above 0.",
                         "Alert VllmPreempting whenever vllm:num_preemptions_total has increased in the last 5 minutes."],
      """- alert: VllmPreempting
  expr: rate(vllm:num_preemptions_total[5m]) > 0
  labels:
    severity: warning
  annotations:
    summary: "vLLM on {{ $labels.instance }} is preempting requests\"""",
      [('vllm:num_preemptions_total{instance="a:8000"}', "0x5 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15")],
      [("4m", False), ("9m", True)], "rate() of a counter is above 0 only while it is increasing."),
    A("VllmSlowFirstToken", ["Rule VllmSlowFirstToken: the 95th percentile of vllm:time_to_first_token_seconds_bucket over 5 minutes above 2 seconds, for 5 minutes.",
                             "Alert me (VllmSlowFirstToken) if p95 time to first token from the vllm:time_to_first_token_seconds histogram exceeds 2s for 5m."],
      """- alert: VllmSlowFirstToken
  expr: histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m]))) > 2
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "vLLM p95 TTFT is {{ $value }}s\"""",
      [('vllm:time_to_first_token_seconds_bucket{le="1"}', "0+10x30"),
       ('vllm:time_to_first_token_seconds_bucket{le="5"}', "0+10x10 100+10x20"),
       ('vllm:time_to_first_token_seconds_bucket{le="+Inf"}', "0+10x10 100+100x20")],
      [("8m", False), ("22m", True)], "histogram_quantile over the summed bucket rates gives the p95."),
    A("PrometheusSelfDown", ["Write PrometheusSelfDown: fire when the prometheus job's scrape_duration_seconds series is missing entirely for 5 minutes.",
                             "Alert PrometheusSelfDown if no scrape_duration_seconds{job=\"prometheus\"} series exists at all for 5m."],
      """- alert: PrometheusSelfDown
  expr: absent(scrape_duration_seconds{job="prometheus"})
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "No scrape_duration_seconds series for the prometheus job\"""",
      [('scrape_duration_seconds{job="prometheus",instance="localhost:9090"}', "0.01x5 _x25")], [("8m", False), ("22m", True)],
      "absent() returns 1 only when no matching series exists."),
    A("MemoryClockDropped", ["Alert MemoryClockDropped: gpulab_gpu_clock_memory_mhz under 1000 for 20 minutes.",
                             "Create a MemoryClockDropped rule for a VRAM clock (gpulab_gpu_clock_memory_mhz) below 1000 MHz that lasts 20m."],
      """- alert: MemoryClockDropped
  expr: gpulab_gpu_clock_memory_mhz < 1000
  for: 20m
  labels:
    severity: info
  annotations:
    summary: "{{ $labels.node }} memory clock at {{ $value }} MHz\"""",
      [('gpulab_gpu_clock_memory_mhz{node="laptop"}', "9000x5 405x30")], [("20m", False), ("30m", True)],
      "20 minutes of `for` filters idle down-clocking."),
    A("GPUMemBusSaturated", ["Write GPUMemBusSaturated: max_over_time of gpulab_gpu_memory_utilization_percent across 10 minutes above 95, firing right away.",
                             "Alert GPUMemBusSaturated as soon as the 10-minute max of gpulab_gpu_memory_utilization_percent is over 95."],
      """- alert: GPUMemBusSaturated
  expr: max_over_time(gpulab_gpu_memory_utilization_percent[10m]) > 95
  labels:
    severity: info
  annotations:
    summary: "{{ $labels.node }} memory bus hit {{ $value }}%\"""",
      [('gpulab_gpu_memory_utilization_percent{node="desktop"}', "40x8 99 40x10")], [("7m", False), ("10m", True)],
      "max_over_time keeps a spike visible for the whole 10-minute window."),
    A("CPUTempRising", ["Rule CPUTempRising: fire when the laptop CPU package temperature (gpulab_cpu_temperature_celsius, sensor=\"package\") rose by more than 20 in the last 10 minutes.",
                        "I need CPUTempRising: alert when delta of gpulab_cpu_temperature_celsius{sensor=\"package\"} over 10m is greater than 20."],
      """- alert: CPUTempRising
  expr: delta(gpulab_cpu_temperature_celsius{sensor="package"}[10m]) > 20
  labels:
    severity: warning
  annotations:
    summary: "{{ $labels.node }} CPU package rose {{ $value }} C in 10m\"""",
      [('gpulab_cpu_temperature_celsius{node="laptop",sensor="package"}', "40x10 45 50 55 60 65 70 75 80 85 90")],
      [("9m", False), ("18m", True)], "delta() is the change of a gauge across the window."),
    A("TooFewGPUs", ["Alert TooFewGPUs: fire when fewer than 2 gpulab_gpu_scrape_ok series exist, for 5 minutes.",
                     "Write TooFewGPUs: count(gpulab_gpu_scrape_ok) under 2 for 5m, i.e. a node's exporter vanished."],
      """- alert: TooFewGPUs
  expr: count(gpulab_gpu_scrape_ok) < 2 or absent(gpulab_gpu_scrape_ok)
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Fewer than 2 GPU exporters reporting\"""",
      [('gpulab_gpu_scrape_ok{node="desktop"}', "1x30"), ('gpulab_gpu_scrape_ok{node="laptop"}', "1x5 _x25")],
      [("8m", False), ("22m", True)], "The `or absent(...)` also covers both exporters vanishing."),
]
OPENERS = ["Here's the rule:", "This rule does it:", "Rule file:", "Drop this into your rules file:",
           "Here is the alerting rule:", "Use this:"]


def eval_items():
    items = {}
    for tag in EVAL_TAGS:
        for mode in ("think-4k", "nothink"):
            p = os.path.join(REPO, "results", f"research-eval-{tag}-lightning-{mode}.json")
            for r in json.load(open(p))["results"]:
                items.setdefault(r["id"], {k: v for k, v in r.items() if k not in ("run", "score")})
    return list(items.values())


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def metrics_in(text):
    return set(re.findall(r"\b(?:gpulab_\w+|vllm:\w+|llamaswap_\w+|node_\w+|scrape_\w+|up)\b", text))


def promql_tool():
    names = rev.urllib.request.urlopen(PROM + "/api/v1/label/__name__/values", timeout=15)
    names = [n for n in json.load(names)["data"] if n.startswith(("gpulab_", "vllm:"))
             and not n.endswith(("_bucket", "_created", "_sum", "_count"))]
    t = json.loads(json.dumps(rev.PROMQL_TOOL))
    t["function"]["description"] += " Metrics: " + ", ".join(sorted(names)) + "."
    return t


def main():
    fail = []
    g6p = json.load(open(G6P))
    items = eval_items()
    eval_qs = sorted({i["question"] for i in items})
    promql_eval_metrics = set().union(*(metrics_in(q) for _, _, q, _, _ in rev.PROMQL_ITEMS))
    alert_eval_metrics = set().union(*(metrics_in(sr) for _, _, ser, _ in rev.ALERTS for sr, _ in ser))
    alert_eval_names = {n for n, *_ in rev.ALERTS}
    tool = promql_tool()
    rng = random.Random(SEED)

    new = []
    for n, s in enumerate(PSPECS):
        for q in s["queries"]:
            shared = metrics_in(q) & promql_eval_metrics
            if shared:
                fail.append(f"promql spec {n}: query {q!r} shares {sorted(shared)} with a promql eval item")
        outs, res = [], []
        for q in s["queries"]:
            out, rec = rev.execute("promql", {"query": q}, set(), WINDOW, PROM)
            if rec.get("outcome") != "executed":
                fail.append(f"promql spec {n}: {q!r} -> {rec}")
            r, _ = rev.prom_query(PROM, q)
            outs.append(out)
            res.append([(m["metric"], m["value"][1]) for m in (r or [])])
        try:
            answer = s["answer"](res)
        except Exception as e:  # a spec whose live data does not have the expected shape
            fail.append(f"promql spec {n}: answer failed on live data {res}: {e!r}")
            continue
        if not any(res) and not rev.NODATA.search(answer):
            fail.append(f"promql spec {n}: every query is empty but the answer misses research_eval's NODATA")
        for q in s["qs"]:
            msgs = [{"role": "user", "content": q}]
            for qq, o in zip(s["queries"], outs):
                msgs.append({"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": "promql", "arguments": json.dumps({"query": qq})}}]})
                msgs.append({"role": "tool", "name": "promql", "content": o})
            msgs.append({"role": "assistant", "content": answer})
            new.append({"type": "promql_live", "tool": "promql", "commands": s["queries"], "spec": n,
                        "second_call": len(s["queries"]) == 2, "extra_tools": [tool], "messages": msgs})

    alert_ok = 0
    for n, s in enumerate(ASPECS):
        shared = metrics_in(s["rule"]) & alert_eval_metrics
        if shared:
            fail.append(f"alert spec {n}: rule shares {sorted(shared)} with an alert eval item")
        if s["name"] in alert_eval_names:
            fail.append(f"alert spec {n}: alertname {s['name']} is an eval alertname")
        answer_body = f"```yaml\ngroups:\n  - name: lab\n    rules:\n" + "\n".join(
            "      " + ln for ln in s["rule"].splitlines()) + "\n```\n\n" + s["note"]
        sc = rev.score_alert({"alertname": s["name"], "series": s["series"], "checks": s["checks"]}, answer_body)
        if not (sc["valid"] and sc["correct"]):
            fail.append(f"alert spec {n} ({s['name']}): score_alert {sc}")
        else:
            alert_ok += 1
        for q in s["qs"]:
            new.append({"type": "alert_direct", "tool": None, "spec": n, "alertname": s["name"],
                        "answer_body": answer_body, "messages": [{"role": "user", "content": q}]})

    alerts = [r for r in new if r["type"] == "alert_direct"]
    order = OPENERS[:]
    rng.shuffle(order)
    for k, i in enumerate(rng.sample(range(len(alerts)), len(alerts))):
        r = alerts[i]
        r["opener"] = order[k % len(order)]
        r["messages"].append({"role": "assistant", "content": r["opener"] + "\n\n" + r.pop("answer_body")})
    off = set(rng.sample(range(len(new)), round(len(new) / 3)))
    for i, r in enumerate(new):
        r["thinking"] = "off" if i in off else "default"
        if "Based on `" in r["messages"][-1]["content"] or rev.GLOBAL_DENIAL.search(r["messages"][-1]["content"]):
            fail.append(f"record {i}: banned opener or denial wording")

    jac = []
    for i, rec in enumerate(new):
        a = words(rec["messages"][0]["content"])
        best = max(((len(a & words(q)) / len(a | words(q)), q) for q in eval_qs))
        jac.append((best[0], i, best[1]))
    jmax = max(jac)
    if jmax[0] >= 0.5:
        fail.append(f"contamination: Jaccard {jmax[0]:.3f} between record {jmax[1]} and eval prompt {jmax[2]!r}")
    prompts = [r["messages"][0]["content"] for r in new]
    if len(set(prompts)) != len(prompts):
        fail.append("duplicate user prompts")

    out = g6p + new
    if json.dumps(out[:len(g6p)]) != json.dumps(g6p):
        fail.append("g6p records changed")
    audit = {
        "g6p_records": len(g6p), "new_records": len(new), "total": len(out),
        "promql_specs": len(PSPECS), "promql_records": sum(r["type"] == "promql_live" for r in new),
        "promql_two_call_records": sum(r.get("second_call", False) for r in new),
        "alert_specs": len(ASPECS), "alert_records": len(alerts), "alert_specs_scored_correct": f"{alert_ok}/{len(ASPECS)}",
        "openers": dict(collections.Counter(r["opener"] for r in alerts)),
        "thinking": dict(collections.Counter(r["thinking"] for r in new)),
        "promql_eval_metrics_excluded": sorted(promql_eval_metrics),
        "alert_eval_metrics_excluded": sorted(alert_eval_metrics),
        "promql_tool_description_chars": len(tool["function"]["description"]),
        "jaccard_max": round(jmax[0], 3), "jaccard_max_pair": {"record": prompts[jmax[1]], "eval": jmax[2]},
        "jaccard_top5": [(round(j, 3), prompts[i], q) for j, i, q in sorted(jac, reverse=True)[:5]],
        "eval_items_checked": len(items), "eval_prompts_checked": len(eval_qs),
        "failures": fail,
    }
    json.dump(audit, open(AUDIT, "w"), indent=1)
    print(json.dumps({k: v for k, v in audit.items() if k not in ("jaccard_top5",)}, indent=1))
    if fail:
        print(f"\nABORT: {len(fail)} failure(s); {OUT} not written", file=sys.stderr)
        sys.exit(1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {len(out)} records ({len(g6p)} g6p + {len(new)} new) -> {OUT}")


if __name__ == "__main__":
    main()
