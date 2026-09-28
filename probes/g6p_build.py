"""G6p (plan.md "G6p"): research_dataset_v3 unchanged, plus task-shaped records whose
answers are full procedures.

Why: G6 and G6r lose rocky_task (and task, thinking off) with one-flag template answers
("Based on `x --help`, the option ...") or a false "does not have X" denial. v3 teaches
that shape: 544 of its 950 answers contain "Based on `". These records ask for an
operator goal, look the tool up, and answer with a complete runnable command or 2-4
numbered steps, one short reason each.

Hand-written task specs (SPECS below), as v3's templates are. Every record is checked
here, and the build aborts on any failure:
  - tools: v3's tools and other subcommands of v3's binaries (docker, git, cargo), none
    named by any eval list (lead, 2026-09-28: option a). Help text is captured on this
    host with gpu-lab's own dataset_generator.run_cmd and cut by its
    get_observation_for_flag, exactly as v3's was, run from this repository's root;
  - grounded: every flag on every answer line passes bench/research_eval.py's
    grounded_token against the captured text (not the fuller help + man page the eval
    grounds against), and no answer line names a flag only to deny it;
  - hit: every flag the spec lists appears in the answer (research_eval's combined
    short-flag reading, as the task scorer uses);
  - no denial: research_eval's GLOBAL_DENIAL never matches an answer;
  - second call: in two-call records, the tokens in `needs` are absent from the first
    output and present in the second, so the first help really lacks the feature;
  - openers: no "Based on `" or "I checked `" anywhere; every opener template on at
    most 15% of the new records;
  - contamination: no new record has the same (binary, flag set) as any eval item in
    any set (flags compared up to the item's aliases); word-set Jaccard between every
    new prompt and every eval prompt < 0.5. Eval items are read from the base runs in
    results/ (research-eval-<set>-lightning-think-4k.json), i.e. exactly what was asked;
  - v3 unchanged: the first 950 output records serialise identically to v3.
Not checked here: token length. probes/g6p_fit.py renders the output with Lightning's
tokenizer as g6_train.py does and must report 0 new records over 1024 tokens.

Thinking: a third of the new records "off" (v3: 257 of 772 tool-using records), seeded.

usage: python3 probes/g6p_build.py   (host python3, no dependency beyond ~/gpu-lab)
writes results/research_dataset_g6p.json and results/g6p-build.json (the audit).
"""
import collections
import json
import os
import random
import re
import statistics
import sys

sys.dont_write_bytecode = True  # nothing written into ~/gpu-lab
GPULAB = os.path.expanduser("~/gpu-lab")
sys.path.insert(0, os.path.join(GPULAB, "training"))
sys.path.insert(0, os.path.join(GPULAB, "bench"))
import dataset_generator as dg  # noqa: E402
import research_eval as rev  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(GPULAB, "training", "research_dataset_v3.json")
OUT = os.path.join(REPO, "results", "research_dataset_g6p.json")
AUDIT = os.path.join(REPO, "results", "g6p-build.json")
EVAL_TAGS = ["v1", "v2", "rocky", "promqlcat", "general", "alert", "trap3"]
SEED = 20260928
EXTRA_TWO_CALL, EXTRA_ONE_CALL = 15, 15
V3_BINARIES = {"docker", "git", "cargo"}

SINGLE_OPENERS = ["Run this:", "One command does it:", "Use this:", "This command does it:",
                  "Here is the command:", "Here's how:", "Try this:", "The command is:"]
STEP_OPENERS = ["Do it in {n} steps:", "Here's the procedure:", "Follow these steps:",
                "The full sequence is:", "Step by step:"]
NUM = {2: "two", 3: "three", 4: "four"}


def S(calls, flags, qs, steps, needs=()):
    return {"calls": calls, "flags": flags, "qs": qs, "steps": steps, "needs": list(needs)}


# (help commands, flags the answer must carry, user goals, steps (command, reason), and
# for two-call specs the tokens the first help lacks and the second has). A one-step
# answer is a command in a code block and a sentence of reasons; more steps are numbered.
SPECS = [
    # ---------------- docker logs
    S(["docker logs --help"], ["--since", "--tail", "-t"],
      ["The api container started throwing 500s about twenty minutes ago. I want only the last 200 lines from the past half hour, each with a timestamp, so I can line them up against the load balancer log.",
       "Grab me the api container's recent output for an incident review: last 200 lines, nothing older than 30 minutes, timestamps on every line."],
      [("docker logs --since 30m --tail 200 -t api",
        "`--since 30m` drops anything older than half an hour, `--tail 200` keeps the last 200 of those lines, and `-t` puts a timestamp on each so they line up with other logs.")]),
    S(["docker logs --help"], ["-f", "-n"],
      ["I want to watch the worker container's output live as it happens, but starting from its latest 20 lines instead of everything it has printed since boot.",
       "How can I keep a live view of what the worker container prints, without first scrolling through its whole history?"],
      [("docker logs -f -n 20 worker",
        "`-f` keeps the stream open and prints new lines as they arrive, and `-n 20` starts from the last 20 lines rather than the whole history.")]),
    S(["docker logs --help"], ["--since", "--until"],
      ["Something broke in the billing container last night between 02:00 and 02:15 UTC on 2026-09-27. Pull out just the log lines from that quarter hour.",
       "For a postmortem I need the billing container's log restricted to the window 2026-09-27 02:00 to 02:15 UTC. Nothing before or after."],
      [("docker logs --since 2026-09-27T02:00:00Z --until 2026-09-27T02:15:00Z billing",
        "`--since` and `--until` both take absolute timestamps in this form, so only lines logged inside the window are printed.")]),
    S(["docker logs --help"], ["--since", "-t"],
      ["Save the last hour of the proxy container's output, stderr included, into a file I can attach to a support ticket.",
       "Our vendor wants a file with the proxy container's logs from the past 60 minutes, timestamps kept. What do I run?"],
      [("docker logs --since 1h -t proxy > proxy-last-hour.log 2>&1",
        "`--since 1h` limits it to the last hour, `-t` keeps the timestamps, and `2>&1` captures what the container wrote to stderr as well as stdout.")]),
    # ---------------- docker ps
    S(["docker ps --help"], ["-l", "--format"],
      ["Which container did I create most recently, even if it has already exited? I only want its ID, name and status, not the full table.",
       "My last docker run seems to have died instantly. How do I get the ID, name and state of the newest container, whatever state it is in?"],
      [("docker ps -l --format '{{.ID}} {{.Names}} {{.Status}}'",
        "`-l` picks the latest created container in any state, and `--format` with a Go template prints only its ID, name and status.")]),
    S(["docker ps --help"], ["--filter", "--format"],
      ["For a shell script I need the names of all running containers that were started from the nginx image, one name per line.",
       "Print just the names of every running nginx-based container so I can loop over them in bash."],
      [("docker ps --filter ancestor=nginx --format '{{.Names}}'",
        "`--filter ancestor=nginx` keeps only containers created from that image, and `--format '{{.Names}}'` prints one bare name per line.")]),
    S(["docker ps --help"], ["-s", "--format"],
      ["The disk on this host keeps filling up and I suspect a container is writing lots of data into its own layer. How do I see how much each running container has written?",
       "Show me, per running container, how many bytes it has written on top of its image, in a small two-column table."],
      [("docker ps -s --format 'table {{.Names}}\\t{{.Size}}'",
        "`-s` adds the size column (what the container wrote, then the virtual total including the image), and the `table` template keeps only names and sizes.")]),
    S(["docker ps --help"], ["--no-trunc", "--format"],
      ["The container list cuts off the command column with an ellipsis. I need the full ID and the complete command line of every running container.",
       "How do I see the untruncated command each running container was started with, next to its full ID?"],
      [("docker ps --no-trunc --format 'table {{.ID}}\\t{{.Command}}'",
        "`--no-trunc` stops Docker shortening IDs and commands, and the template limits the table to those two columns.")]),
    S(["docker ps --help"], ["-n", "--format"],
      ["Show the three containers I created most recently, whatever state they are in, with their exit status so I can see which one crashed.",
       "I launched three test containers in a row and some exited. List exactly those three newest ones with their status."],
      [("docker ps -n 3 --format 'table {{.Names}}\\t{{.Status}}'",
        "`-n 3` lists the three most recently created containers in any state, and the Status column reads `Exited (code)` for the ones that stopped.")]),
    S(["docker ps --help"], ["-q", "--filter"],
      ["I label containers with env=staging. How do I get only the IDs of the running ones with that label, for piping into another command?",
       "Output nothing but the container IDs of running containers labelled env=staging."],
      [("docker ps -q --filter label=env=staging",
        "`--filter label=env=staging` keeps containers carrying that label, and `-q` prints only their IDs, one per line.")]),
    # docker ps -> sibling subcommand (the first help lacks the action)
    S(["docker ps --help", "docker stop --help"], ["-q", "--filter", "-t"],
      ["Stop every running container that was started from redis:7, and give each one 30 seconds to shut down cleanly before Docker kills it.",
       "We are retiring redis:7 on this host. Bring down all of its running containers gracefully, with a 30 second grace period each."],
      [("docker stop -t 30 $(docker ps -q --filter ancestor=redis:7)",
        "The inner `docker ps -q --filter ancestor=redis:7` prints the IDs of the matching containers, and `docker stop -t 30` gives each of them 30 seconds before it is killed.")],
      needs=["-t", "--timeout"]),
    S(["docker ps --help", "docker restart --help"], ["-q", "--filter", "-t"],
      ["Restart all running containers labelled tier=web, letting each take up to 20 seconds to stop first.",
       "After rotating a certificate I need every tier=web container restarted, with a 20 second grace period for each."],
      [("docker restart -t 20 $(docker ps -q --filter label=tier=web)",
        "`docker ps -q --filter label=tier=web` supplies the IDs, and `docker restart -t 20` waits up to 20 seconds for each to stop before starting it again.")],
      needs=["restart", "--timeout"]),
    S(["docker ps --help", "docker kill --help"], ["-q", "--filter", "-s"],
      ["Send SIGHUP to every running nginx container so they reload their configuration without being restarted.",
       "Tell all the nginx containers on this box to re-read their config by signalling them, not by restarting them."],
      [("docker kill -s HUP $(docker ps -q --filter ancestor=nginx)",
        "`docker kill -s HUP` sends a hangup signal instead of the default SIGKILL, and the inner `docker ps -q --filter ancestor=nginx` supplies the IDs.")],
      needs=["signal"]),
    S(["docker ps --help", "docker port --help"], ["--filter", "--format"],
      ["I started a postgres container with random published ports and forgot which host port maps to 5432. How do I find it?",
       "Which host port is Docker forwarding to port 5432 in my running postgres container?"],
      [("docker ps --filter ancestor=postgres --format '{{.Names}}'", "finds the name of the running postgres container"),
       ("docker port NAME 5432/tcp", "prints the host address and port mapped to 5432, with NAME being the name from step 1")],
      needs=["mapping"]),
    # ---------------- docker stats / top
    S(["docker stats --help", "docker top --help"], ["--no-stream"],
      ["The scraper container is pinning a CPU core. How do I confirm that and then see which process inside it is responsible, without opening a shell in it?",
       "Find out which process in the scraper container is burning CPU, from the host."],
      [("docker stats --no-stream scraper", "takes one reading of the container's CPU and memory, so you can confirm it is the culprit"),
       ("docker top scraper aux", "lists every process inside it with its CPU and memory share; `aux` is handed straight to ps")],
      needs=["processes"]),
    S(["docker stats --help"], ["--no-stream", "--format"],
      ["I want a one-off snapshot of CPU and memory for every running container, not the live-updating screen, so I can paste it into a ticket.",
       "Print container resource usage once and exit, showing only name, CPU percentage and memory used."],
      [("docker stats --no-stream --format 'table {{.Name}}\\t{{.CPUPerc}}\\t{{.MemUsage}}'",
        "`--no-stream` prints one reading and exits instead of refreshing, and the template keeps the name, CPU and memory columns.")]),
    S(["docker stats --help"], ["-a", "--no-stream", "--no-trunc"],
      ["Show resource usage for every container on the host, stopped ones included, with full container IDs, as a single reading.",
       "I need a one-time usage table covering all containers, not just running ones, and the IDs must not be shortened."],
      [("docker stats -a --no-stream --no-trunc",
        "`-a` includes stopped containers, `--no-stream` takes a single reading, and `--no-trunc` prints full IDs.")]),
    S(["docker stats --help"], ["--format"],
      ["Keep a live eye on memory for just the api and worker containers: name, memory used and memory percentage.",
       "Watch only two containers, api and worker, showing their memory usage live."],
      [("docker stats --format 'table {{.Name}}\\t{{.MemUsage}}\\t{{.MemPerc}}' api worker",
        "Naming api and worker limits the view to those two, and the template shows memory used and memory percentage, refreshing live.")]),
    # ---------------- docker stop / kill / restart / start / rm / pause / wait / rename / update
    S(["docker stop --help"], ["-s", "-t"],
      ["The app container shuts down cleanly on SIGINT but not on SIGTERM, and it needs up to a minute. Stop it that way.",
       "Stop the app container by sending SIGINT rather than SIGTERM, and allow it 60 seconds before it gets killed."],
      [("docker stop -s SIGINT -t 60 app",
        "`-s SIGINT` sends the signal the app handles, and `-t 60` waits 60 seconds for a clean exit before Docker falls back to SIGKILL.")]),
    S(["docker stop --help"], ["-t"],
      ["Our database container needs up to two minutes to flush to disk on shutdown. Stop it without Docker killing it after the usual 10 seconds.",
       "How do I stop the db container while giving it 120 seconds to finish writing?"],
      [("docker stop -t 120 db",
        "`-t 120` makes Docker wait 120 seconds for a clean exit before it sends SIGKILL.")]),
    S(["docker stop --help", "docker rm --help"], ["-t", "-v"],
      ["I want the old-api container gone for good: stop it gracefully first, then delete it together with the anonymous volumes it created.",
       "Retire the old-api container: clean shutdown, then remove the container and its anonymous volumes."],
      [("docker stop -t 30 old-api", "lets it shut down cleanly, with 30 seconds before a kill"),
       ("docker rm -v old-api", "deletes the container and the anonymous volumes it created")],
      needs=["--volumes"]),
    S(["docker rm --help"], ["-f", "-v"],
      ["A container named tmp-builder is wedged in a running state and I just want it deleted, volumes included, in one command.",
       "Delete the running tmp-builder container right now, along with its anonymous volumes."],
      [("docker rm -f -v tmp-builder",
        "`-f` kills it with SIGKILL if it is still running, and `-v` removes its anonymous volumes with it.")]),
    S(["docker kill --help"], ["-s"],
      ["The encoder container ignores SIGTERM and I need it down right now, but I want it to get SIGINT rather than a hard kill.",
       "Stop the encoder container immediately by sending it SIGINT."],
      [("docker kill -s SIGINT encoder",
        "`docker kill` sends the signal at once, and `-s SIGINT` picks SIGINT instead of the default SIGKILL.")]),
    S(["docker stop --help", "docker kill --help"], ["-s"],
      ["The ingest container rotates its logs when it receives SIGUSR1. How do I send it that signal without stopping it?",
       "Deliver SIGUSR1 to the running ingest container and leave it running afterwards."],
      [("docker kill -s SIGUSR1 ingest",
        "`docker kill -s SIGUSR1` delivers only that signal, so a process that handles SIGUSR1 keeps running.")],
      needs=["kill"]),
    S(["docker restart --help"], ["-s", "-t"],
      ["I changed a config file on the host. Restart the gateway container, but it drains connections on SIGQUIT and needs 45 seconds for that.",
       "Restart gateway so it gets SIGQUIT and has 45 seconds to drain before being killed."],
      [("docker restart -s SIGQUIT -t 45 gateway",
        "`-s SIGQUIT` sends the signal that makes it drain, and `-t 45` waits 45 seconds before killing and starting it again.")]),
    S(["docker start --help"], ["-a", "-i"],
      ["My devbox container exited. Start it again and put me straight back into it, with its output on my terminal and my keyboard connected.",
       "How do I restart a stopped interactive container named devbox and be attached to it?"],
      [("docker start -a -i devbox",
        "`-a` attaches its output to your terminal and forwards signals, and `-i` connects your keyboard to its STDIN.")]),
    S(["docker update --help", "docker rename --help"], [],
      ["The container I launched is called nostalgic_hopper. Change its name to metrics-db without recreating it.",
       "How do I give an existing container, nostalgic_hopper, the new name metrics-db while keeping everything else?"],
      [("docker rename nostalgic_hopper metrics-db",
        "This renames the container in place; it keeps its ID, and a running container keeps running.")],
      needs=["rename"]),
    S(["docker update --help"], ["--memory", "--memory-swap", "--cpus"],
      ["Limit the running etl container to 2 GB of memory with no extra swap and one and a half CPUs, without restarting it.",
       "The etl container is hogging the host. Cap it at 2 GB RAM, no swap on top, and 1.5 CPUs while it keeps running."],
      [("docker update --memory 2g --memory-swap 2g --cpus 1.5 etl",
        "`--memory 2g` caps RAM, `--memory-swap 2g` set to the same value leaves no swap on top, and `--cpus 1.5` limits CPU time; the change applies to the running container.")]),
    S(["docker update --help"], ["--restart"],
      ["Make the cache container come back by itself after a crash or a host reboot, unless I stopped it on purpose.",
       "I forgot to set a restart policy on the cache container. Set one now so it survives reboots but stays down if I stop it."],
      [("docker update --restart unless-stopped cache",
        "`--restart unless-stopped` restarts it after crashes and daemon restarts, but not after you stop it yourself; no need to recreate the container.")]),
    S(["docker update --help"], ["--pids-limit", "--cpu-shares"],
      ["The crawler container forks too many processes. Cap it at 200 processes and give it half the default CPU weight.",
       "Rein in the crawler container: at most 200 PIDs, and lower CPU priority than the other containers."],
      [("docker update --pids-limit 200 --cpu-shares 512 crawler",
        "`--pids-limit 200` stops it creating more than 200 processes, and `--cpu-shares 512` halves its weight against the default 1024 when CPUs are contended.")]),
    S(["docker stop --help", "docker pause --help"], [],
      ["I need to freeze the batch container for a few minutes while I snapshot the disk, then let it carry on exactly where it was.",
       "How do I suspend every process in the batch container temporarily, without stopping it, and resume later?"],
      [("docker pause batch", "freezes all its processes in place; memory and state are kept"),
       ("docker unpause batch", "resumes them where they stopped once the snapshot is done")],
      needs=["pause"]),
    S(["docker wait --help"], [],
      ["In a CI script I start a migration container. How do I block until it finishes and then get its exit code?",
       "Make my deploy script wait for the migrate container to stop and capture its exit status."],
      [("rc=$(docker wait migrate)",
        "`docker wait` blocks until the container stops and prints its exit code, which this stores in `rc` for the script to test.")]),
    # ---------------- docker exec / cp / attach
    S(["docker exec --help"], ["-i", "-t", "-u"],
      ["Open a root shell inside the running web container, even though its image runs as an unprivileged user.",
       "I need to poke around the web container as root. It normally runs as a non-root user."],
      [("docker exec -it -u root web sh",
        "`-i` keeps STDIN open and `-t` allocates a terminal so the shell is usable, `-u root` overrides the image's user, and `sh` exists even in images without bash.")]),
    S(["docker exec --help"], ["-w", "-e"],
      ["Run the Rails migrations inside the running app container, from /srv/app, with RAILS_ENV=production set.",
       "Execute bin/rails db:migrate in the app container in its /srv/app directory with the production environment variable."],
      [("docker exec -w /srv/app -e RAILS_ENV=production app bin/rails db:migrate",
        "`-w /srv/app` sets the working directory for the command, and `-e RAILS_ENV=production` sets the variable only for this run.")]),
    S(["docker exec --help"], ["-d"],
      ["Kick off the long cache warm-up script inside the running cache container without tying up my terminal.",
       "Start /usr/local/bin/warm-cache in the cache container in the background and get my prompt back."],
      [("docker exec -d cache /usr/local/bin/warm-cache",
        "`-d` runs the command in the background inside the container and returns immediately.")]),
    S(["docker exec --help"], ["--env-file"],
      ["Run a one-off Django check in the worker container with every variable from my local .env.prod file set.",
       "How do I pass all the settings in .env.prod into a single command I run inside the worker container?"],
      [("docker exec --env-file .env.prod worker python manage.py check",
        "`--env-file .env.prod` reads the file on the host and sets each variable for this one command.")]),
    S(["docker exec --help", "docker cp --help"], [],
      ["The api container left a crash dump at /tmp/core.1234. I need that file on my host to open in gdb.",
       "Get /tmp/core.1234 out of the api container and onto my local disk."],
      [("docker cp api:/tmp/core.1234 ./core.1234",
        "This copies the file from the container to the current directory, and it works whether or not the container is still running.")],
      needs=["copy"]),
    S(["docker cp --help", "docker kill --help"], ["-s"],
      ["I edited nginx.conf on the host. Put it into the running proxy container at /etc/nginx/nginx.conf and make nginx pick it up without a restart.",
       "Push my local nginx.conf into the proxy container and have nginx reload it live."],
      [("docker cp ./nginx.conf proxy:/etc/nginx/nginx.conf", "puts the edited file in place inside the container"),
       ("docker kill -s HUP proxy", "sends nginx the hangup signal, which makes it reload its configuration without restarting")],
      needs=["signal"]),
    S(["docker cp --help"], ["-a"],
      ["Back up the /var/lib/app/data directory from the app container to the host, keeping the owner and group of every file.",
       "Copy the data directory out of the app container with file ownership preserved."],
      [("docker cp -a app:/var/lib/app/data ./data-backup",
        "`-a` keeps the uid and gid of every file, and a directory source is copied with everything under it.")]),
    S(["docker cp --help"], ["-L"],
      ["In the app container /etc/app is a symlink. I want the real files it points to copied to my host, not the link itself.",
       "Copying /etc/app out of the container gave me a dangling symlink. How do I get the actual contents?"],
      [("docker cp -L app:/etc/app ./app-config",
        "`-L` follows the symlink in the source path and copies what it points to.")]),
    S(["docker attach --help"], ["--no-stdin", "--sig-proxy"],
      ["I want to watch the output of the running app container live, but my keystrokes and Ctrl-C must not reach its process.",
       "Attach to the app container's output read-only, so pressing Ctrl-C cannot kill it."],
      [("docker attach --no-stdin --sig-proxy=false app",
        "`--no-stdin` leaves its input unattached, and `--sig-proxy=false` stops Ctrl-C and other signals being passed to the process.")]),
    # ---------------- docker inspect / events / info / logs -> inspect
    S(["docker inspect --help"], ["-f"],
      ["What IP address does the db container have on its Docker network? Just the address, not the whole JSON document.",
       "Print only the network IP of the db container."],
      [("docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' db",
        "`-f` applies a Go template to the JSON, and ranging over the attached networks prints the IP on each.")]),
    S(["docker inspect --help"], ["-f"],
      ["The worker container stopped. Show me its restart policy and the exit code it stopped with, on one line.",
       "Why won't worker come back? Print its restart policy name and its last exit code."],
      [("docker inspect -f '{{.HostConfig.RestartPolicy.Name}} {{.State.ExitCode}}' worker",
        "`-f` pulls just those two fields out of the container's JSON.")]),
    S(["docker inspect --help"], ["-s", "-f"],
      ["How much disk is the scratch container using? I want the bytes it wrote itself and the total including its image.",
       "Get the writable-layer size and the full size of the scratch container from docker inspect."],
      [("docker inspect -s -f '{{.SizeRw}} {{.SizeRootFs}}' scratch",
        "`-s` adds the size fields for a container: `SizeRw` is what it wrote, `SizeRootFs` includes the image.")]),
    S(["docker inspect --help"], ["--type"],
      ["I have an image and a container both called cache. How do I inspect only the image?",
       "docker inspect cache shows me the container, but I want the image of the same name."],
      [("docker inspect --type image cache",
        "`--type image` makes Docker look up only images, so the container with the same name is ignored.")]),
    S(["docker logs --help", "docker inspect --help"], ["--tail", "-f"],
      ["The worker container keeps restarting. I want its last output before it died and how many times it has restarted, with the exit code.",
       "Work out why worker is in a restart loop: its final log lines plus its restart count and last exit code."],
      [("docker logs --tail 50 worker", "shows the last 50 lines it printed before the latest exit"),
       ("docker inspect -f '{{.RestartCount}} {{.State.ExitCode}}' worker", "prints how often it has been restarted and the exit code of the last run")],
      needs=["--format"]),
    S(["docker events --help"], ["--since", "--filter"],
      ["Show every container that died on this host in the last 10 minutes and keep watching for new ones.",
       "I suspect containers are crashing intermittently. Stream container die events, starting 10 minutes back."],
      [("docker events --since 10m --filter event=die",
        "`--since 10m` replays the last ten minutes first, and `--filter event=die` keeps only containers exiting; it keeps streaming until you stop it.")]),
    S(["docker events --help"], ["--since", "--until", "--filter", "--format"],
      ["Give me, as JSON, the container events of the past hour and then exit instead of streaming forever.",
       "For a report I need the last hour of container events in JSON, printed once."],
      [("docker events --since 1h --until \"$(date +%s)\" --filter type=container --format json",
        "`--since 1h` starts an hour back, `--until` set to now makes it exit, `--filter type=container` drops image and network events, and `--format json` prints one JSON object per event.")]),
    S(["docker info --help"], ["-f"],
      ["Which storage driver and cgroup version is this Docker host using? Just those two values.",
       "Print only the storage driver and the cgroup version from docker info."],
      [("docker info -f '{{.Driver}} {{.CgroupVersion}}'",
        "`-f` applies a Go template to the info output, so only those two fields are printed.")]),
    S(["docker login --help"], ["-u", "--password-stdin"],
      ["Log in to registry.example.com from a CI job with a token in $REGISTRY_TOKEN, without the token showing up in the process list.",
       "How should a pipeline authenticate to registry.example.com as user ci-bot, reading the password from an environment variable safely?"],
      [("echo \"$REGISTRY_TOKEN\" | docker login -u ci-bot --password-stdin registry.example.com",
        "`-u ci-bot` sets the user, and `--password-stdin` reads the token from the pipe, so it never appears as a command-line argument.")]),
    # ---------------- docker images / rmi / tag / pull / push / save / load / history / commit / export / import
    S(["docker images --help"], ["-q", "--filter"],
      ["List only the IDs of dangling images, the untagged <none> ones, so I can feed them to another command.",
       "Print the image IDs of all untagged leftover images on this host, IDs only."],
      [("docker images -q --filter dangling=true",
        "`--filter dangling=true` keeps untagged images, and `-q` prints only their IDs.")]),
    S(["docker images --help", "docker rmi --help"], ["-q", "--filter"],
      ["Delete every dangling image on this build box to get disk space back.",
       "Our CI host is full of untagged <none> images. Remove all of them."],
      [("docker rmi $(docker images -q --filter dangling=true)",
        "The inner command prints the IDs of all untagged images, and `docker rmi` removes them.")],
      needs=["remove"]),
    S(["docker images --help"], ["--digests", "--no-trunc"],
      ["Show the full sha256 digest and full image ID of every local version of registry.example.com/api.",
       "I need to pin a deployment by digest. List the local registry.example.com/api images with complete digests and IDs."],
      [("docker images --digests --no-trunc registry.example.com/api",
        "`--digests` adds the digest column, `--no-trunc` prints full IDs, and naming the repository limits the list to it.")]),
    S(["docker images --help"], ["--format"],
      ["Print local images as a table of repository, tag and size only.",
       "Give me a compact image list: name, tag and how big each image is."],
      [("docker images --format 'table {{.Repository}}\\t{{.Tag}}\\t{{.Size}}'",
        "`--format` with a `table` template prints just those three columns with headers.")]),
    S(["docker images --help"], ["--filter", "-q"],
      ["Which local images were built before python:3.12? Give me their IDs for cleanup.",
       "List the IDs of images older than my python:3.12 image."],
      [("docker images -q --filter before=python:3.12",
        "`--filter before=python:3.12` keeps images created before that one, and `-q` prints only their IDs.")]),
    S(["docker images --help", "docker tag --help"], [],
      ["Give our local build myapp:latest a second name, registry.example.com/team/myapp:1.4.0, so I can push it.",
       "How do I add the name registry.example.com/team/myapp:1.4.0 to the image I built locally as myapp:latest?"],
      [("docker images myapp", "confirms the local image and shows its ID"),
       ("docker tag myapp:latest registry.example.com/team/myapp:1.4.0", "adds the new name to the same image; nothing is copied")],
      needs=["TARGET_IMAGE"]),
    S(["docker rmi --help"], ["-f", "--no-prune"],
      ["docker rmi refuses to delete old-image:2.1 because a stopped container still uses it. Remove it anyway, but keep its untagged parent layers.",
       "Force-remove the image old-image:2.1 while leaving its untagged parents in place."],
      [("docker rmi -f --no-prune old-image:2.1",
        "`-f` removes it although a stopped container references it, and `--no-prune` keeps its untagged parent images.")]),
    S(["docker rmi --help"], ["--platform"],
      ["I pulled a multi-platform alpine:3.20 and only want to drop its linux/arm64 variant, keeping the amd64 one.",
       "Remove just the arm64 variant of alpine:3.20 from the local store."],
      [("docker rmi --platform linux/arm64 alpine:3.20",
        "`--platform linux/arm64` removes only that variant and leaves the others.")]),
    S(["docker pull --help"], ["--platform", "-q"],
      ["On my x86 laptop I need the arm64 build of ubuntu:24.04 to test a Raspberry Pi deployment, and I don't want progress bars in the CI log.",
       "Fetch the linux/arm64 variant of ubuntu:24.04 quietly on an amd64 machine."],
      [("docker pull --platform linux/arm64 -q ubuntu:24.04",
        "`--platform linux/arm64` selects that variant even on an amd64 host, and `-q` suppresses the progress output.")]),
    S(["docker pull --help"], ["-a"],
      ["Download every tag of registry.example.com/tools/lint in one go, for an offline mirror.",
       "Mirror all tagged versions of the tools/lint image from registry.example.com."],
      [("docker pull -a registry.example.com/tools/lint",
        "`-a` downloads every tagged image in the repository instead of just `latest`.")]),
    S(["docker push --help"], ["-a", "-q"],
      ["Push all tags of registry.example.com/team/myapp in one command, without the progress noise.",
       "Upload every tag I have locally for registry.example.com/team/myapp, quietly."],
      [("docker push -a -q registry.example.com/team/myapp",
        "`-a` pushes every tag of the repository, and `-q` keeps the output short.")]),
    S(["docker tag --help", "docker push --help"], ["-q"],
      ["I built api:dev locally. Get it into our registry as registry.example.com/api:2.0.",
       "Publish my local image api:dev to registry.example.com under the tag 2.0."],
      [("docker tag api:dev registry.example.com/api:2.0", "gives the image the registry name and tag"),
       ("docker push -q registry.example.com/api:2.0", "uploads it under that name")],
      needs=["upload"]),
    S(["docker pull --help", "docker tag --help"], ["-q"],
      ["Mirror nginx:1.27 into our registry namespace: fetch it and give it the name registry.example.com/mirror/nginx:1.27.",
       "Pull nginx:1.27 quietly and retag it as registry.example.com/mirror/nginx:1.27."],
      [("docker pull -q nginx:1.27", "fetches the image without progress bars"),
       ("docker tag nginx:1.27 registry.example.com/mirror/nginx:1.27", "adds the mirror name to the same image, ready to push")],
      needs=["TARGET_IMAGE"]),
    S(["docker save --help", "docker load --help"], ["-o", "-i"],
      ["Move the image app:1.3 to an air-gapped server: write it to a file here, then load it there.",
       "How do I carry app:1.3 to a host with no registry access, using a file?"],
      [("docker save -o app-1.3.tar app:1.3", "writes the image, all layers and its tag, into one tar file"),
       ("docker load -i app-1.3.tar", "on the other host reads it back with the same name and tag")],
      needs=["load"]),
    S(["docker save --help"], ["--platform", "-o"],
      ["Export only the linux/amd64 variant of api:2.0 to a tar file for a customer who runs x86 servers.",
       "Save api:2.0 to a file, but just its amd64 variant."],
      [("docker save --platform linux/amd64 -o api-amd64.tar api:2.0",
        "`--platform linux/amd64` keeps only that variant, and `-o` writes to the file instead of stdout.")]),
    S(["docker load --help"], ["-q", "-i"],
      ["A colleague sent me images.tar.gz from docker save. Load it without the progress output.",
       "Import the saved images in images.tar.gz into my local Docker, quietly."],
      [("docker load -q -i images.tar.gz",
        "`-i` reads the archive from the file (gzip is fine), and `-q` suppresses the load output.")]),
    S(["docker history --help"], ["--no-trunc", "--format"],
      ["Which layer made the ml-base image so big? I want each layer's size next to the full command that created it.",
       "Break down ml-base by layer: size and the complete Dockerfile step for each."],
      [("docker history --no-trunc --format 'table {{.Size}}\\t{{.CreatedBy}}' ml-base",
        "`--no-trunc` shows each creating command in full, and the template keeps only size and command.")]),
    S(["docker commit --help"], ["-a", "-m"],
      ["Snapshot the current state of the debug container as an image, with an author and a message, so a colleague can reproduce the bug.",
       "Save the debug container's filesystem as debug-snapshot:812, recording who made it and why."],
      [("docker commit -a \"Dana <dana@example.com>\" -m \"state after repro of issue 812\" debug debug-snapshot:812",
        "`-a` records the author and `-m` the message on the new image, which is written as debug-snapshot:812.")]),
    S(["docker commit --help"], ["-c"],
      ["Turn the sandbox container into an image that starts /bin/bash by default.",
       "Create an image from sandbox whose default command is bash."],
      [("docker commit -c 'CMD [\"/bin/bash\"]' sandbox sandbox:shell",
        "`-c` applies a Dockerfile instruction to the new image, here setting its default command.")]),
    S(["docker commit --help", "docker push --help"], ["-m"],
      ["Save the debug container as an image and publish it to registry.example.com as debug:812 for the on-call engineer.",
       "Get the current state of the debug container into our registry under registry.example.com/debug:812."],
      [("docker commit -m \"repro for 812\" debug registry.example.com/debug:812", "creates the image straight under its registry name"),
       ("docker push registry.example.com/debug:812", "uploads it so the on-call engineer can pull it")],
      needs=["upload"]),
    S(["docker export --help", "docker import --help"], ["-o", "-m"],
      ["Flatten the builder container into a single-layer image called builder-flat:1.",
       "I want builder's filesystem as a one-layer image named builder-flat:1, dropping the layer history."],
      [("docker export -o builder.tar builder", "writes the container's whole filesystem as one tar file"),
       ("docker import -m \"flattened from builder\" builder.tar builder-flat:1", "turns that file into a single-layer image")],
      needs=["import"]),
    S(["docker start --help", "docker logs --help"], ["-f", "-n"],
      ["Start the stopped nightly container again and follow only what it prints from now on.",
       "Bring the nightly container back up and watch its new output live, skipping the old lines."],
      [("docker start nightly", "starts the existing container in the background"),
       ("docker logs -f -n 0 nightly", "follows its output, and `-n 0` skips everything logged before now")],
      needs=["--follow"]),
    # ---------------- git log / show / shortlog / reflog / describe
    S(["git log -h"], ["-L"],
      ["Show every commit that touched lines 40 to 80 of src/parser.rs, with the diff of just those lines.",
       "Trace how the block at src/parser.rs lines 40-80 changed over time."],
      [("git log -L 40,80:src/parser.rs",
        "`-L 40,80:src/parser.rs` follows that line range back through history and shows each commit's change to it.")]),
    S(["git log -h"], ["-L"],
      ["I want the history of the function parse_header in lib/http.c: every commit that changed it.",
       "Follow the evolution of one function, parse_header in lib/http.c, through the commit history."],
      [("git log -L :parse_header:lib/http.c",
        "`-L :parse_header:lib/http.c` finds the function by name and shows every commit that changed its body.")]),
    S(["git show -h"], ["-q"],
      ["Show the tag message and commit details of v2.0 without the whole diff.",
       "What does the v2.0 tag say, and which commit is it on? I don't need the patch."],
      [("git show -q v2.0",
        "`-q` suppresses the diff, leaving the tag's message and the commit header.")]),
    S(["git log -h", "git shortlog -h"], ["-s", "-n", "-e"],
      ["Who has committed to this repo and how many commits each, busiest first, with email addresses?",
       "I need a contributor list for this repository: commit count per author, sorted, emails included."],
      [("git shortlog -s -n -e HEAD",
        "`-s` gives one count per author, `-n` sorts by that count, `-e` adds emails, and naming HEAD makes it read the history instead of waiting on stdin.")],
      needs=["-n"]),
    S(["git log -h", "git shortlog -h"], ["-s", "-c"],
      ["Count the commits per committer, not per author, made since the v2.0 tag.",
       "Since v2.0, how many commits did each committer land?"],
      [("git shortlog -s -n -c v2.0..HEAD",
        "`-c` groups by committer instead of author, `-s -n` prints sorted counts only, and `v2.0..HEAD` limits it to commits after the tag.")],
      needs=["-c"]),
    S(["git log -h", "git reflog -h"], [],
      ["I ran a hard reset and my last commit disappeared from the branch. How do I get it back?",
       "A git reset --hard threw away a commit I needed. Recover it."],
      [("git reflog", "lists every position HEAD has been at, newest first; the lost commit is the entry just before the reset"),
       ("git branch rescue HEAD@{1}", "puts a branch on that entry (use the number you found) so the commit is safe again")],
      needs=["reflog"]),
    S(["git describe -h"], ["--tags", "--dirty", "--always"],
      ["For a build version string, which tag is the current commit closest to, and is the working tree modified? It must print something even with no tags.",
       "Produce a version label from git: nearest tag, commits since, a marker if uncommitted changes exist, and a fallback to the hash."],
      [("git describe --tags --dirty --always",
        "`--tags` accepts lightweight tags too, `--dirty` appends `-dirty` when the tree has changes, and `--always` falls back to the short hash if no tag is reachable.")]),
    S(["git describe -h"], ["--contains"],
      ["Which release first shipped commit 4b7e2d1? I want the earliest tag that contains it.",
       "Find the first tag that includes commit 4b7e2d1."],
      [("git describe --contains 4b7e2d1",
        "`--contains` names the tag that comes after the commit, i.e. the first one that includes it.")]),
    S(["git describe -h"], ["--tags", "--match", "--abbrev"],
      ["What is the most recent tag starting with v that HEAD descends from? Just the tag name, nothing appended.",
       "Print the latest v-prefixed tag reachable from HEAD, bare."],
      [("git describe --tags --match 'v*' --abbrev=0",
        "`--match 'v*'` only considers tags starting with v, `--tags` includes lightweight ones, and `--abbrev=0` prints the bare tag without the commit count and hash.")]),
    # ---------------- git status
    S(["git status -h"], ["-s", "-b"],
      ["Give me a compact status: one line per changed file plus the branch and how far ahead or behind upstream it is.",
       "Short git status that still shows my branch and its ahead/behind counts."],
      [("git status -s -b",
        "`-s` prints one line per file, and `-b` adds a header line with the branch and its ahead/behind counts.")]),
    S(["git status -h"], ["--porcelain", "-b", "-z"],
      ["A script needs git status in a machine-readable format that stays stable across git versions, with NUL separators for odd filenames.",
       "Stable, parseable status output for tooling, including branch info, safe for paths with spaces or newlines."],
      [("git status --porcelain=v2 -b -z",
        "`--porcelain=v2` is the stable machine format, `-b` adds branch headers, and `-z` ends entries with NUL so any filename parses.")]),
    S(["git status -h"], ["--ignored", "--untracked-files"],
      ["Show me ignored files too, and list every untracked file individually instead of just their directories.",
       "I want status to include files matched by .gitignore and to expand untracked directories into their files."],
      [("git status --ignored --untracked-files=all",
        "`--ignored` adds the ignored files, and `--untracked-files=all` lists each untracked file rather than its folder.")]),
    S(["git status -h"], ["--untracked-files", "--ignore-submodules"],
      ["git status takes ages in this huge monorepo with many untracked files and submodules. Make it fast.",
       "Speed up git status here by skipping untracked files and submodule checks."],
      [("git status --ignore-submodules=all --untracked-files=no",
        "`--untracked-files=no` skips scanning for untracked files, and `--ignore-submodules=all` skips checking submodules.")]),
    # ---------------- git reset
    S(["git reset -h"], ["--soft"],
      ["Squash my last three local commits into a single commit, keeping all their changes.",
       "Combine the three most recent unpushed commits into one."],
      [("git reset --soft HEAD~3", "moves the branch back three commits and leaves all their changes staged"),
       ("git commit", "records the staged changes as one commit; your editor opens for the message")]),
    S(["git reset -h"], ["-q", "--mixed"],
      ["Unstage everything I added, but keep all the edits in my files.",
       "I staged too much. Empty the index without touching my working copy."],
      [("git reset -q --mixed HEAD",
        "`--mixed` resets the index to HEAD but leaves the working tree alone, and `-q` keeps it quiet.")]),
    S(["git reset -h"], ["--hard"],
      ["Throw away my local commits and uncommitted changes so my branch matches origin/main exactly.",
       "Make my checkout identical to what is on origin/main, discarding everything local."],
      [("git fetch origin", "updates origin/main to what the server has"),
       ("git reset --hard origin/main", "points the branch there and overwrites the index and working tree")]),
    S(["git reset -h"], ["--keep"],
      ["Move my branch back one commit, but fail instead of overwriting any uncommitted changes I have.",
       "Drop the last commit from my branch while protecting my local edits."],
      [("git reset --keep HEAD~1",
        "`--keep` moves the branch back one commit and refuses if that would overwrite your local changes.")]),
    S(["git reset -h"], ["-p"],
      ["I staged all of src/app.py but only want some of its hunks in the next commit. Unstage the rest interactively.",
       "Pick hunk by hunk what to take back out of the index for src/app.py."],
      [("git reset -p src/app.py",
        "`-p` walks through the staged hunks of that file and asks which to unstage.")]),
    S(["git reset -h"], ["--pathspec-from-file"],
      ["I have a list of 300 paths in paths.txt that I staged by mistake. Unstage exactly those.",
       "Unstage every file listed in paths.txt."],
      [("git reset --pathspec-from-file=paths.txt",
        "`--pathspec-from-file` reads the paths from the file, one per line, and unstages only those.")]),
    # ---------------- git diff
    S(["git diff -h"], ["--name-status"],
      ["Which files changed between the v1.2 tag and HEAD, and was each added, modified or deleted?",
       "List the files touched since v1.2 with their change type."],
      [("git diff --name-status v1.2 HEAD",
        "`--name-status` prints each changed file with A, M or D instead of the full patch.")]),
    S(["git diff -h"], ["--cached", "--stat"],
      ["How big are my staged changes, per file, before I commit?",
       "Show a per-file summary of what is staged right now."],
      [("git diff --cached --stat",
        "`--cached` compares the index with HEAD, and `--stat` shows lines added and removed per file.")]),
    S(["git diff -h"], ["--text"],
      ["git diff says 'Binary files differ' for our minified bundle dist/app.min.js. Make it show the actual changes.",
       "Force a text diff of dist/app.min.js."],
      [("git diff --text dist/app.min.js",
        "`--text` treats the file as text, so the changed lines are shown.")]),
    # ---------------- git stash
    S(["git stash -h"], ["-u", "-m"],
      ["I need to switch branches but have half-done work, including brand-new untracked files. Put all of it aside with a note.",
       "Stash everything, new files included, with a description so I can find it later."],
      [("git stash push -u -m \"wip: parser refactor\"",
        "`-u` includes untracked files, and `-m` labels the entry so it is easy to find in `git stash list`.")]),
    S(["git stash -h"], ["-m"],
      ["Stash only my changes under config/ and leave the rest of the working tree alone.",
       "Set aside just the config/ directory's modifications."],
      [("git stash push -m \"config tweaks\" -- config/",
        "The pathspec after `--` limits the stash to config/, and `-m` labels it.")]),
    S(["git stash -h"], ["--index"],
      ["Bring back my latest stash, and restore what was staged as staged, not just as working-tree changes.",
       "Pop the newest stash and keep the staged/unstaged split it had."],
      [("git stash pop --index",
        "`--index` restores the staged state too, and `pop` drops the entry once it applies cleanly.")]),
    S(["git stash -h"], [],
      ["An old stash no longer applies cleanly on my branch. Turn stash@{2} into its own branch instead.",
       "Recover the third stash entry onto a fresh branch where it applies without conflicts."],
      [("git stash list", "confirms stash@{2} is the entry you want"),
       ("git stash branch fix-from-stash stash@{2}", "creates a branch at the commit the stash was made on and applies it there")]),
    S(["git stash -h"], ["--staged", "-m"],
      ["Stash only what I have staged and keep my unstaged edits in the working tree.",
       "Put aside the staged part of my changes and nothing else."],
      [("git stash push --staged -m \"staged part\"",
        "`--staged` stashes only the index, leaving unstaged changes where they are.")]),
    # ---------------- git clean / rm / mv
    S(["git clean -h"], ["-n", "-d", "-f"],
      ["Delete all untracked files and directories in this repo, but let me see what would go first.",
       "Clear out untracked junk, folders included, after a preview."],
      [("git clean -n -d", "lists every untracked file and directory that would be removed"),
       ("git clean -f -d", "removes them; `-f` is required, and `-d` includes directories")]),
    S(["git clean -h"], ["-n", "-X", "-f"],
      ["Remove only the ignored build output, like target/ and *.o files, and keep my untracked source files.",
       "Wipe ignored artefacts but not the new files I haven't added yet."],
      [("git clean -n -d -X", "previews the ignored files and directories that would go"),
       ("git clean -f -d -X", "deletes only those; `-X` restricts it to ignored paths")]),
    S(["git clean -h"], ["-f", "-x", "-e"],
      ["Reset the repo to a pristine checkout: remove untracked and ignored files, but keep my .env file.",
       "Remove everything git doesn't track, ignored files too, except .env."],
      [("git clean -f -d -x -e .env",
        "`-x` removes ignored files as well, `-d` includes directories, and `-e .env` adds an exception so .env stays.")]),
    S(["git clean -h", "git rm -h"], ["-r", "--cached", "-f"],
      ["Our build/ directory is partly tracked by mistake and partly untracked. Get rid of all of it, from the repo and from disk.",
       "Remove build/ completely: stop tracking the files that were committed and delete the untracked rest."],
      [("git rm -r --cached build/", "stops tracking the committed files without touching disk"),
       ("git clean -f -d -x build/", "then deletes everything left in build/")],
      needs=["--cached"]),
    S(["git rm -h"], ["--cached"],
      ["I committed .env by accident. Stop tracking it but keep the file on my disk.",
       "Remove .env from the repository while leaving my local copy."],
      [("git rm --cached .env", "removes it from the index only; the file stays on disk"),
       ("echo .env >> .gitignore", "keeps it from being added again")]),
    S(["git rm -h"], ["-r", "--cached", "--ignore-unmatch"],
      ["In a cleanup script, remove the whole vendor/ directory from the index, keep the files, and don't fail if it's already gone.",
       "Untrack vendor/ recursively without deleting it, safely for repeated runs."],
      [("git rm -r --cached --ignore-unmatch vendor/",
        "`-r` recurses into the directory, `--cached` leaves the files on disk, and `--ignore-unmatch` exits 0 when nothing matches.")]),
    S(["git mv -h"], ["-n", "-v", "-f"],
      ["Rename src/Utils.py to src/utils.py on a case-insensitive filesystem. Show me what happens first.",
       "Change only the case of a tracked filename, src/Utils.py to src/utils.py, with a dry run first."],
      [("git mv -n -v src/Utils.py src/utils.py", "shows the rename it would do"),
       ("git mv -f src/Utils.py src/utils.py", "does it; `-f` lets it proceed where the filesystem sees the names as the same file")]),
    # ---------------- git worktree
    S(["git worktree -h"], ["-b"],
      ["Start a fix from origin/main in a separate directory ../fix-123 on a new branch, without stashing my current work.",
       "Create a new branch fix-123 from origin/main checked out in ../fix-123, leaving this checkout untouched."],
      [("git worktree add -b fix-123 ../fix-123 origin/main",
        "`-b fix-123` creates the branch from origin/main, and it is checked out in ../fix-123 beside your current work.")]),
    S(["git worktree -h"], ["-n", "-v"],
      ["I deleted a worktree folder by hand and git still thinks it exists. Clean up its records, showing what gets removed.",
       "Remove stale worktree entries after I rm -rf'd the directory, with a preview."],
      [("git worktree prune -n -v", "shows the stale worktree entries"),
       ("git worktree prune -v", "removes them and reports each one")]),
    S(["git worktree -h"], ["--porcelain"],
      ["List my worktrees in a format a script can parse.",
       "Machine-readable list of all worktrees of this repo."],
      [("git worktree list --porcelain",
        "`--porcelain` prints one attribute per line with a blank line between worktrees, stable for scripts.")]),
    S(["git worktree -h"], ["-f"],
      ["Remove the worktree at ../exp even though it still has uncommitted changes I don't need.",
       "Delete the ../exp worktree and throw away its local modifications."],
      [("git worktree remove -f ../exp",
        "`-f` removes it even with uncommitted changes.")]),
    S(["git worktree -h"], ["--reason"],
      ["One worktree lives on a USB drive. Stop git from pruning it while the drive is unplugged.",
       "Protect the worktree at /media/usb/proj from being pruned, with a note why."],
      [("git worktree lock --reason \"on usb drive\" /media/usb/proj",
        "`lock` keeps it from being pruned or moved, and `--reason` records why.")]),
    # ---------------- git bisect
    S(["git bisect -h"], ["--first-parent"],
      ["Find which merge into main between v3.1 (good) and HEAD (broken) broke the build, testing automatically with ./test.sh.",
       "Automate a bisect over main's merges from v3.1 to HEAD using ./test.sh as the check."],
      [("git bisect start --first-parent HEAD v3.1", "marks HEAD bad and v3.1 good, stepping only along main's merges"),
       ("git bisect run ./test.sh", "runs the script at each step: exit 0 marks good, 125 skips, any other code up to 127 marks bad"),
       ("git bisect reset", "returns you to where you started")]),
    S(["git bisect -h"], ["--no-checkout"],
      ["Bisect a huge repo between v1.0 and HEAD without checking out each commit, because my test only reads objects.",
       "Start a bisection that doesn't touch the working tree at every step."],
      [("git bisect start --no-checkout HEAD v1.0",
        "`--no-checkout` moves only the BISECT_HEAD ref, so nothing is checked out; mark each step with `git bisect good` or `git bisect bad`.")]),
    # ---------------- git remote
    S(["git remote -h"], ["-v"],
      ["The repo moved. Rename the old remote origin to upstream, add my fork as origin, and check the result.",
       "Make the original repository upstream and my fork origin."],
      [("git remote rename origin upstream", "keeps the original repository under the name upstream"),
       ("git remote add origin git@github.com:me/proj.git", "adds your fork as origin"),
       ("git remote -v", "shows both with their URLs")]),
    S(["git remote -h"], ["-v"],
      ["Our server changed its address. Point origin at git@git.example.com:team/proj.git and confirm it.",
       "Update the origin URL to the new SSH address and verify."],
      [("git remote set-url origin git@git.example.com:team/proj.git", "replaces the URL"),
       ("git remote -v", "shows the new fetch and push URLs")]),
    S(["git remote -h"], ["-n"],
      ["Delete my remote-tracking branches that no longer exist on origin, but preview the list first.",
       "Clean up stale origin/ branches that were deleted on the server, dry run first."],
      [("git remote prune -n origin", "lists the stale remote-tracking branches"),
       ("git remote prune origin", "deletes them")]),
    S(["git remote -h"], ["-t", "-f"],
      ["Add a remote called vendor that only tracks its main branch, and fetch it immediately.",
       "Add https://example.com/vendor/lib.git as vendor, tracking main only, and download it now."],
      [("git remote add -t main -f vendor https://example.com/vendor/lib.git",
        "`-t main` tracks only that branch, and `-f` fetches right after adding.")]),
    S(["git remote -h"], ["--add", "--push"],
      ["I want one git push to go to both GitHub and our GitLab mirror.",
       "Make origin push to two servers at once."],
      [("git remote set-url --add --push origin git@github.com:team/proj.git", "adds GitHub as a push URL; once any is set only push URLs are used, so the original goes in too"),
       ("git remote set-url --add --push origin git@gitlab.example.com:team/proj.git", "adds the mirror, so each push goes to both")]),
    # ---------------- git revert / cherry-pick
    S(["git revert -h"], ["-m"],
      ["A bad merge commit 3c9e1f2 is already on main. Undo it, keeping main's side of the merge.",
       "Revert merge commit 3c9e1f2 on a shared branch."],
      [("git revert -m 1 3c9e1f2",
        "`-m 1` tells git the first parent, main, is the line to keep, and the revert is a new commit so history is not rewritten.")]),
    S(["git revert -h"], ["-n"],
      ["Revert the last three commits as one single new commit.",
       "Undo HEAD~2, HEAD~1 and HEAD together in one revert commit."],
      [("git revert -n HEAD~3..HEAD", "applies all three reversals to the index without committing"),
       ("git commit", "records them as one revert commit")]),
    S(["git cherry-pick -h"], ["-x"],
      ["Copy commit 4f2a9c1 from main onto my release branch, recording where it came from in the message.",
       "Backport 4f2a9c1 to the current branch with a reference to the original commit."],
      [("git cherry-pick -x 4f2a9c1",
        "`-x` appends \"(cherry picked from commit ...)\" to the message, so the origin is recorded.")]),
    S(["git cherry-pick -h"], ["-n"],
      ["Apply commits a1b2c3d and e4f5a6b to my branch but combine them into one commit.",
       "Bring two commits over without committing them separately."],
      [("git cherry-pick -n a1b2c3d e4f5a6b", "applies both to the index without committing"),
       ("git commit", "records them as one commit")]),
    # ---------------- git switch / restore
    S(["git switch -h"], ["-c", "--track"],
      ["Create a branch feature/login from origin/main, switch to it, and have it track origin/main.",
       "New branch feature/login based on origin/main with upstream set, checked out."],
      [("git switch -c feature/login --track origin/main",
        "`-c` creates and switches to the branch, and `--track` sets origin/main as its upstream.")]),
    S(["git switch -h"], ["--detach"],
      ["I want to look at the code as it was at tag v2.0 without creating a branch.",
       "Check out v2.0 read-only style, no new branch."],
      [("git switch --detach v2.0",
        "`--detach` puts HEAD directly on the tag's commit; switching back later leaves no branch behind.")]),
    S(["git switch -h"], ["--discard-changes"],
      ["Switch to main and throw away my local modifications in the process.",
       "Go to main, dropping whatever I changed here."],
      [("git switch --discard-changes main",
        "`--discard-changes` discards local modifications so the switch cannot fail on them.")]),
    S(["git switch -h"], ["--orphan"],
      ["Start an empty gh-pages branch with no history for our static site.",
       "Create a branch that shares no commits with the rest of the repo."],
      [("git switch --orphan gh-pages",
        "`--orphan` creates the branch with no history and an empty working tree.")]),
    S(["git restore -h"], ["--staged"],
      ["Unstage src/db.py but keep my edits in the file.",
       "Take src/db.py out of the index without losing the changes."],
      [("git restore --staged src/db.py",
        "`--staged` restores the index entry from HEAD and leaves the working copy alone.")]),
    S(["git restore -h"], ["--source"],
      ["Bring back the version of config.yml from two commits ago into my working copy.",
       "Replace config.yml with how it looked at HEAD~2."],
      [("git restore --source=HEAD~2 config.yml",
        "`--source=HEAD~2` takes the file from that commit and writes it to the working tree.")]),
    S(["git restore -h"], ["--theirs"],
      ["During a merge conflict in package-lock.json I just want the incoming branch's version.",
       "Resolve the package-lock.json conflict by taking their side."],
      [("git restore --theirs package-lock.json", "writes their version into the working tree"),
       ("git add package-lock.json", "marks the conflict as resolved")]),
    S(["git restore -h"], ["--staged", "--worktree", "--source"],
      ["Reset both the staged and the working copy of README.md back to HEAD.",
       "Discard every change to README.md, staged or not."],
      [("git restore --staged --worktree --source=HEAD README.md",
        "`--staged` and `--worktree` restore both copies, and `--source=HEAD` takes them from the last commit.")]),
    # ---------------- git gc / archive / init / submodule / notes / add
    S(["git gc -h"], ["--aggressive", "--prune"],
      ["The repo's .git is huge after a big history rewrite. Repack thoroughly and drop unreachable objects now.",
       "Shrink .git as much as possible, pruning unreferenced objects immediately."],
      [("git gc --aggressive --prune=now",
        "`--aggressive` repacks more thoroughly, and `--prune=now` deletes unreachable objects without the usual grace period.")]),
    S(["git gc -h", "git reflog -h"], ["--expire", "--all", "--prune"],
      ["After removing a secret from history, make sure the old objects are really gone from this clone.",
       "Expire all reflog entries and garbage-collect so rewritten-away commits are purged."],
      [("git reflog expire --expire=now --all", "drops every reflog entry, which still points at the old commits"),
       ("git gc --prune=now", "then deletes the objects nothing references any more")],
      needs=["--expire"]),
    S(["git archive -h"], ["--format", "--prefix", "-o"],
      ["Make a release tarball of tag v1.4.0 with everything under a proj-1.4.0/ folder.",
       "Build proj-1.4.0.tar.gz from the v1.4.0 tag, files nested under proj-1.4.0/."],
      [("git archive --format=tar.gz --prefix=proj-1.4.0/ -o proj-1.4.0.tar.gz v1.4.0",
        "`--format=tar.gz` compresses it, `--prefix` puts every file under proj-1.4.0/, and `-o` names the output file.")]),
    S(["git archive -h"], ["--format", "-o"],
      ["Zip up only the docs/ directory as it is on main.",
       "Give me docs.zip containing docs/ from the main branch."],
      [("git archive --format=zip -o docs.zip main docs/",
        "`--format=zip` makes a zip, `-o` names it, and the trailing `docs/` limits the archive to that directory.")]),
    S(["git init -h"], ["--bare", "-b"],
      ["Create a bare repository on the server at /srv/git/proj.git for people to push to, with main as the default branch.",
       "Set up a push target repo on our server with main as the initial branch."],
      [("git init --bare -b main /srv/git/proj.git",
        "`--bare` creates a repository with no working tree, and `-b main` names the initial branch.")]),
    S(["git init -h"], ["--separate-git-dir"],
      ["Initialise a repo in the current folder but keep the .git data on another disk at /data/git/proj.git.",
       "New repository here, with its object store living on /data."],
      [("git init --separate-git-dir /data/git/proj.git .",
        "`--separate-git-dir` puts the repository there and leaves a .git file here pointing to it.")]),
    S(["git submodule -h"], ["--init", "--recursive"],
      ["After cloning, fetch all submodules, and their own submodules, at the commits the superproject records.",
       "My clone has empty submodule folders. Populate them, nested ones included."],
      [("git submodule update --init --recursive",
        "`--init` registers and clones any uninitialised submodule, and `--recursive` repeats it inside nested ones.")]),
    S(["git submodule -h"], ["--remote", "--merge"],
      ["Move every submodule to the latest commit of the branch it tracks, merging into any local work there.",
       "Update submodules to their upstream branch tips instead of the recorded commits."],
      [("git submodule update --remote --merge",
        "`--remote` uses each submodule's remote-tracking branch, and `--merge` merges it into what is checked out.")]),
    S(["git notes -h"], ["-m"],
      ["Attach a note to commit 4d1e7a2 saying it was deployed to prod on 2026-09-28, without changing the commit.",
       "Record deployment info on commit 4d1e7a2 without rewriting it."],
      [("git notes add -m \"deployed to prod 2026-09-28\" 4d1e7a2",
        "Notes are stored beside the commit, so its hash stays the same; `-m` gives the text.")]),
    S(["git add -h"], ["-n", "-A"],
      ["Stage everything, new files and deletions included, after previewing what gets added.",
       "Add all changes to the index, dry run first."],
      [("git add -n -A", "lists what would be staged"),
       ("git add -A", "stages new, modified and deleted files")]),
    # ---------------- cargo
    S(["cargo clean --help"], ["-n", "-r"],
      ["Free disk by removing only the release build artifacts of this Rust project, but show what would be deleted first.",
       "Clear target/release while keeping the debug build, with a preview."],
      [("cargo clean -n -r", "shows what would be deleted"),
       ("cargo clean -r", "removes only the release artifacts")]),
    S(["cargo clean --help"], ["--doc"],
      ["Delete just the generated documentation, not the compiled build.",
       "Wipe target/doc and nothing else."],
      [("cargo clean --doc",
        "`--doc` cleans only the documentation directory, so the build cache stays.")]),
    S(["cargo clean --help"], ["-p"],
      ["In a workspace, clean only the parser package so it rebuilds from scratch, leaving the rest cached.",
       "Force a full rebuild of one crate, parser, without touching the others."],
      [("cargo clean -p parser",
        "`-p parser` removes only that package's artifacts.")]),
    S(["cargo clean --help"], ["--target"],
      ["Remove only the build output for the aarch64-unknown-linux-gnu target.",
       "Clean the cross-compiled aarch64 artifacts but not the native ones."],
      [("cargo clean --target aarch64-unknown-linux-gnu",
        "`--target` limits the clean to that target triple's output directory.")]),
    S(["cargo clippy --help"], ["--fix"],
      ["Apply clippy's suggested fixes automatically to my crate.",
       "Let clippy rewrite my code with its suggestions."],
      [("cargo clippy --fix",
        "`--fix` applies the suggestions and implies `--no-deps`; it refuses to run on uncommitted changes, so commit first.")]),
    S(["cargo clippy --help"], ["--no-deps", "-D"],
      ["Make our CI job fail on any clippy warning in our own crate.",
       "Turn every clippy warning into an error for CI, ignoring dependencies."],
      [("cargo clippy --no-deps -- -D warnings",
        "`--no-deps` lints only this crate, and `-D warnings` after `--` turns every warning into an error.")]),
    S(["cargo clippy --help"], ["--explain"],
      ["What is clippy's needless_lifetimes lint about?",
       "Show me clippy's own documentation for needless_lifetimes."],
      [("cargo clippy --explain needless_lifetimes",
        "`--explain` prints the documentation for that lint.")]),
    S(["cargo clippy --help"], ["-A", "-W"],
      ["For one run, allow too_many_arguments but warn on the whole pedantic group.",
       "Run clippy with pedantic warnings on and too_many_arguments silenced."],
      [("cargo clippy -- -A clippy::too_many_arguments -W clippy::pedantic",
        "`-A` allows that lint and `-W` warns on the pedantic group, both passed after `--`.")]),
    S(["cargo update --help"], ["--precise"],
      ["Pin serde to exactly 1.0.210 in Cargo.lock without updating anything else.",
       "Set the locked serde version to 1.0.210 only."],
      [("cargo update serde --precise 1.0.210",
        "Naming serde limits the update to it, and `--precise` sets that exact version in the lock file.")]),
    S(["cargo update --help"], ["--dry-run", "--verbose"],
      ["What would cargo update change in Cargo.lock? Show it without writing anything.",
       "Preview dependency updates, with detail, leaving the lock file untouched."],
      [("cargo update --dry-run --verbose",
        "`--dry-run` does not write the lock file, and `--verbose` also lists what stays unchanged.")]),
    S(["cargo update --help"], ["--recursive"],
      ["Update tokio and everything tokio depends on, leaving other crates alone.",
       "Refresh tokio together with its own dependencies."],
      [("cargo update tokio --recursive",
        "Naming tokio selects it, and `--recursive` updates its dependencies too.")]),
    S(["cargo new --help"], ["--lib", "--edition", "--vcs"],
      ["Start a new library crate called netparse on edition 2021, without creating a git repository.",
       "Create netparse as a library crate, 2021 edition, no VCS."],
      [("cargo new --lib --edition 2021 --vcs none netparse",
        "`--lib` uses the library template, `--edition 2021` sets the edition, and `--vcs none` skips git init.")]),
    S(["cargo init --help"], ["--bin", "--name"],
      ["Turn the existing folder ./tools into a binary crate named devtools.",
       "Make ./tools a Rust application whose package is called devtools."],
      [("cargo init --bin --name devtools ./tools",
        "`--bin` uses the application template, and `--name` sets the package name instead of the folder name.")]),
    S(["cargo fmt --help"], ["--all", "--check"],
      ["In CI, check formatting of every package in the workspace without changing any file.",
       "Fail the pipeline if any workspace crate is not rustfmt-clean."],
      [("cargo fmt --all --check",
        "`--all` covers every package, and `--check` reports differences and exits non-zero instead of rewriting files.")]),
    S(["cargo fmt --help"], ["-p"],
      ["Format only the api package in this workspace.",
       "Run rustfmt on just the api crate."],
      [("cargo fmt -p api",
        "`-p api` limits formatting to that package.")]),
    S(["cargo search --help"], ["--limit"],
      ["Find the top 20 crates for async sqlite.",
       "Search crates.io for async sqlite and show 20 results."],
      [("cargo search --limit 20 async sqlite",
        "`--limit 20` returns 20 results instead of the default 10.")]),
    S(["cargo uninstall --help"], ["--root"],
      ["Remove the cargo-watch binary that I installed under /opt/cargo.",
       "Uninstall cargo-watch from the /opt/cargo install root."],
      [("cargo uninstall --root /opt/cargo cargo-watch",
        "`--root` points at the install directory it was put in.")]),
    S(["cargo remove --help"], ["--dev", "-p"],
      ["Drop mockall from the dev-dependencies of the core package in this workspace.",
       "Remove the dev-dependency mockall from core's Cargo.toml."],
      [("cargo remove --dev -p core mockall",
        "`--dev` removes it from dev-dependencies, and `-p core` edits that package's manifest.")]),
    S(["cargo metadata --help"], ["--no-deps", "--format-version"],
      ["Get JSON metadata for the workspace members only, without resolving dependencies.",
       "Machine-readable info about my workspace packages, no dependency graph."],
      [("cargo metadata --no-deps --format-version 1",
        "`--no-deps` covers only workspace members, and `--format-version 1` pins the JSON format.")]),
    S(["cargo vendor --help"], ["--versioned-dirs", "--offline"],
      ["Copy all dependencies into third_party/ so this project builds on a machine with no internet.",
       "Vendor every crate into third_party with versioned folders, then build offline."],
      [("cargo vendor --versioned-dirs third_party", "copies each dependency into third_party/ and prints the [source] config that uses it"),
       ("cargo build --offline", "builds from the vendored copies once that config is in .cargo/config.toml")]),
    # ---------------- free / curl
    S(["free --help"], ["-h", "-s", "-c"],
      ["Watch memory use every 2 seconds, ten times, in human-readable units, then stop.",
       "Sample RAM and swap ten times at 2 second intervals, readable sizes."],
      [("free -h -s 2 -c 10",
        "`-h` picks readable units, `-s 2` repeats every 2 seconds, and `-c 10` stops after ten reports.")]),
    S(["free --help"], ["-g", "-t"],
      ["Show memory in gibibytes, with a line totalling RAM plus swap.",
       "How much RAM plus swap is there in total, in GiB?"],
      [("free -g -t",
        "`-g` reports in gibibytes, and `-t` adds a total line for RAM plus swap.")]),
    S(["free --help"], ["-w", "-h"],
      ["I want buffers and page cache as separate columns in free's output.",
       "Split buff/cache into two columns, human-readable."],
      [("free -w -h",
        "`-w` shows buffers and cache separately, and `-h` picks readable units.")]),
    S(["free --help"], ["-L", "-s"],
      ["Log memory usage on a single line every 60 seconds, for a long-running capture.",
       "One-line memory summary repeated every minute."],
      [("free -L -s 60",
        "`-L` prints everything on one line, and `-s 60` repeats it every 60 seconds.")]),
    S(["free --help"], ["-v", "-h"],
      ["How much memory is committed right now, compared with the commit limit?",
       "Show committed memory and the overcommit limit, readable."],
      [("free -v -h",
        "`-v` adds the committed memory and commit limit, and `-h` picks readable units.")]),
    S(["curl --help"], ["-f", "-O"],
      ["Download https://example.com/releases/tool-1.2.tar.gz keeping its name, and fail rather than saving an HTML error page.",
       "Fetch a release tarball into a file named as on the server, erroring out on HTTP errors."],
      [("curl -f -O https://example.com/releases/tool-1.2.tar.gz",
        "`-O` saves under the remote file name, and `-f` makes an HTTP error exit non-zero with nothing written.")]),
    S(["curl --help"], ["-i", "-u", "-d"],
      ["POST the form field name=node7 to https://api.example.com/nodes with basic auth admin:secret, and show the response headers.",
       "Send a form POST with credentials and see status and headers in the output."],
      [("curl -i -u admin:secret -d 'name=node7' https://api.example.com/nodes",
        "`-d` sends the form data as a POST, `-u` adds basic auth, and `-i` prints the response headers too.")]),
    S(["curl --help"], ["-s", "-f", "-T", "-u"],
      ["Upload backup.tar to https://files.example.com/upload/ with a username and password, silently, failing on errors.",
       "PUT a file to our upload endpoint from cron, quiet unless it fails."],
      [("curl -s -f -T backup.tar -u backup:pw https://files.example.com/upload/",
        "`-T` uploads the file, `-u` authenticates, `-s` silences progress, and `-f` exits non-zero on HTTP errors.")]),
    S(["curl --help"], ["-A", "-o"],
      ["Save https://example.com/ to page.html while identifying as a regular browser.",
       "Fetch a page with a browser User-Agent into a file."],
      [("curl -A \"Mozilla/5.0\" -o page.html https://example.com/",
        "`-A` sets the User-Agent header, and `-o` writes to page.html.")]),
    S(["curl --help"], ["-s", "-f", "-o"],
      ["In a health check script, hit https://svc.example.com/healthz, print nothing, and just exit non-zero if it is unhealthy.",
       "Quiet readiness probe with curl that only signals through the exit code."],
      [("curl -s -f -o /dev/null https://svc.example.com/healthz",
        "`-s` silences progress, `-o /dev/null` discards the body, and `-f` makes an HTTP error return a non-zero exit code.")]),
]


def help_tool(cmd):
    if cmd.startswith("man "):
        return cmd[4:]
    return re.sub(r"\s+(--help|-h)$", "", cmd)


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def has_word(w, text):
    if w.startswith("-"):
        return rev.has_token(w, text)
    return re.search(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])", text, re.I) is not None


def says(x, answer):  # research_eval's task hit, without the docs lookup for combined flags
    if rev.has_token(x, answer):
        return True
    return bool(re.fullmatch(r"-[A-Za-z]", x)) and any(
        x[1] in t[1:] for t in re.findall(r"(?<![\w/.-])-[A-Za-z]{2,6}(?![\w-])", answer))


def render(spec, opener, style):
    steps = spec["steps"]
    if len(steps) == 1:
        cmd, why = steps[0]
        return f"{opener}\n\n```\n{cmd}\n```\n\n{why}"
    sep = ": " if style == 0 else " — "
    lines = [f"{i}. `{cmd}`{sep}{why}." for i, (cmd, why) in enumerate(steps, 1)]
    return opener.format(n=NUM[len(steps)]) + "\n\n" + "\n".join(lines)


def eval_items():
    items = {}
    for tag in EVAL_TAGS:
        for mode in ("think-4k", "nothink"):
            p = os.path.join(REPO, "results", f"research-eval-{tag}-lightning-{mode}.json")
            for r in json.load(open(p))["results"]:
                items.setdefault(r["id"], {k: v for k, v in r.items() if k not in ("run", "score")})
    return list(items.values())


def flag_groups(item):
    if item.get("groups") is not None:
        return [list(g) for g in item["groups"]]
    if item.get("aliases"):
        return [list(item["aliases"])]
    f = item.get("fake_flag") or item.get("flag")
    return [[f]] if f else []


def same_flag_set(flags, groups):
    if not flags and not groups:
        return True
    return (bool(groups) and all(any(a in flags for a in g) for g in groups)
            and all(any(f in g for g in groups) for f in flags))


def main():
    os.chdir(REPO)  # git's -h output differs outside a repository
    fail = []
    v3 = json.load(open(V3))
    v3_tools = {r["tool"] for r in v3 if r.get("tool")}
    items = eval_items()
    eval_tools = {i["tool"] for i in items if i.get("tool")}
    eval_tools |= {t for t, _ in rev.HELD_OUT + rev.HELD_OUT2 + rev.ROCKY_TOOLS}
    eval_tools |= {t[0] for t in rev.TASKS + rev.ROCKY_TASKS + rev.TRAPS3} | set(rev.ROCKY_BINS)
    eval_qs = sorted({i["question"] for i in items})

    captured = {}
    for spec in SPECS:
        for c in spec["calls"]:
            if c not in captured:
                raw = dg.run_cmd(c)
                lines = (raw or "").splitlines()
                if len(lines) < 5:
                    sys.exit(f"ABORT: `{c}` printed no help here: {raw!r}")
                captured[c] = lines

    rng = random.Random(SEED)
    # Every spec's first phrasing; the second for EXTRA_TWO_CALL two-call specs and
    # EXTRA_ONE_CALL one-call specs, drawn with the seed (about 200 records, ~20% two-call).
    two = [n for n, s in enumerate(SPECS) if len(s["calls"]) == 2]
    one = [n for n, s in enumerate(SPECS) if len(s["calls"]) == 1]
    extra = set(rng.sample(two, EXTRA_TWO_CALL)) | set(rng.sample(one, EXTRA_ONE_CALL))
    recs = []
    for n, spec in enumerate(SPECS):
        tools = [help_tool(c) for c in spec["calls"]]
        for t in tools:
            if t in eval_tools:
                fail.append(f"spec {n}: tool {t} is named by an eval list")
            if t not in v3_tools and t.split()[0] not in V3_BINARIES:
                fail.append(f"spec {n}: tool {t} is neither a v3 tool nor a subcommand of {sorted(V3_BINARIES)}")
        anchor = spec["flags"][0] if spec["flags"] else None
        outs = [dg.get_observation_for_flag(captured[c], anchor) for c in spec["calls"]]
        if len(outs) == 2:
            for w in spec["needs"]:
                if has_word(w, outs[0]) or not has_word(w, outs[1]):
                    fail.append(f"spec {n}: needs {w!r} absent from `{spec['calls'][0]}` and present in `{spec['calls'][1]}`")
            if not spec["needs"]:
                fail.append(f"spec {n}: two calls but no needs")
        for q in spec["qs"][:2 if n in extra else 1]:
            recs.append({"spec": n, "tool": tools[-1], "tools": tools, "calls": spec["calls"], "outs": outs,
                         "q": q, "second_call": len(outs) == 2})

    # openers: balanced within each answer shape, then thinking, both seeded
    for shape, openers in ((1, SINGLE_OPENERS), (2, STEP_OPENERS)):
        idx = [i for i, r in enumerate(recs) if (len(SPECS[r["spec"]]["steps"]) == 1) == (shape == 1)]
        order = openers[:]
        rng.shuffle(order)
        for k, i in enumerate(rng.sample(idx, len(idx))):
            recs[i]["opener"] = order[k % len(order)]
            recs[i]["style"] = k % 2
    off = set(rng.sample(range(len(recs)), round(len(recs) / 3)))

    new = []
    grounded_pass = 0
    for i, r in enumerate(recs):
        spec = SPECS[r["spec"]]
        answer = render(spec, r["opener"], r["style"])
        ref = rev.expand_no("\n".join(r["outs"]))
        bad = []
        for line in answer.splitlines():
            for t in rev.FLAG_RE.findall(line):
                if not rev.grounded_token(t, ref):
                    bad.append(t)
            if rev.negated_spans(line):
                bad.append(f"negated: {line}")
        if bad:
            fail.append(f"record {i} (spec {r['spec']}): ungrounded {bad}")
        else:
            grounded_pass += 1
        for f in spec["flags"]:
            if not says(f, answer):
                fail.append(f"record {i}: answer lacks its flag {f}")
            if not rev.grounded_token(f, ref):
                fail.append(f"record {i}: flag {f} not in captured help")
        if rev.GLOBAL_DENIAL.search(answer):
            fail.append(f"record {i}: denial wording: {rev.GLOBAL_DENIAL.search(answer).group(0)!r}")
        if "Based on `" in answer or "I checked `" in answer:
            fail.append(f"record {i}: banned opener")
        msgs = [{"role": "user", "content": r["q"]}]
        for c, o in zip(r["calls"], r["outs"]):
            msgs.append({"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": c})}}]})
            msgs.append({"role": "tool", "name": "bash", "content": o})
        msgs.append({"role": "assistant", "content": answer})
        new.append({"type": "task_procedure", "tool": r["tool"], "tools": r["tools"], "flags": spec["flags"],
                    "commands": r["calls"], "second_call": r["second_call"], "opener": r["opener"],
                    "thinking": "off" if i in off else "default", "messages": msgs})

    # contamination guard
    # The rule: same tool (v3's granularity: "git revert" is not "git rebase") and the same
    # flag set, up to the item's aliases. Reported beside it, not enforced: the same test on
    # the binary alone, which also pairs e.g. `git revert -m` with git rebase's --merge/-m.
    guard_hits, binary_hits = [], []
    for i, rec in enumerate(new):
        for it in items:
            if not it.get("tool"):
                continue
            if it["tool"].split()[0] == rec["tool"].split()[0] and same_flag_set(rec["flags"], flag_groups(it)):
                (guard_hits if it["tool"] == rec["tool"] else binary_hits).append(
                    (i, rec["tool"], rec["flags"], it["id"]))
    for h in guard_hits:
        fail.append(f"contamination: record {h[0]} ({h[1]} {h[2]}) matches eval item {h[3]}")
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

    openers = collections.Counter(r["opener"] for r in new)
    for o, k in openers.items():
        if k > 0.15 * len(new):
            fail.append(f"opener {o!r} on {k} of {len(new)} (> 15%)")
    if len(openers) < 8:
        fail.append(f"only {len(openers)} distinct openers")
    tools = collections.Counter(r["tool"] for r in new)
    if len(tools) < 15:
        fail.append(f"only {len(tools)} tools")

    out = v3 + new
    if json.dumps(out[:len(v3)]) != json.dumps(v3):
        fail.append("v3 records changed")

    second = sum(r["second_call"] for r in new)
    audit = {
        "v3_records": len(v3), "new_records": len(new), "total": len(out), "specs": len(SPECS),
        "tools": len(tools), "tools_v3": sorted(t for t in tools if t in v3_tools),
        "tools_subcommand_not_in_v3": sorted(t for t in tools if t not in v3_tools),
        "tool_counts": dict(sorted(tools.items())),
        "all_called_tools": len({t for r in new for t in r["tools"]}),
        "second_call_records": second, "second_call_share": round(second / len(new), 3),
        "openers": dict(openers), "opener_max_share": round(max(openers.values()) / len(new), 3),
        "one_step_records": sum(len(SPECS[r["spec"]]["steps"]) == 1 for r in recs),
        "multi_step_records": sum(len(SPECS[r["spec"]]["steps"]) > 1 for r in recs),
        "thinking": dict(collections.Counter(r["thinking"] for r in new)),
        "grounded_pass": f"{grounded_pass}/{len(new)}",
        "contamination_same_tool_flag_set_hits": len(guard_hits),
        "tools_shared_with_eval_items": sorted({r["tool"] for r in new} & {i["tool"] for i in items if i.get("tool")}),
        "info_same_binary_flag_set": sorted({(h[1], tuple(h[2]), h[3]) for h in binary_hits}),
        "jaccard_max": round(jmax[0], 3), "jaccard_max_pair": {"record": prompts[jmax[1]], "eval": jmax[2]},
        "jaccard_top5": [(round(j, 3), prompts[i], q) for j, i, q in sorted(jac, reverse=True)[:5]],
        "eval_items_checked": len(items), "eval_prompts_checked": len(eval_qs),
        "based_on_in_new": sum("Based on `" in r["messages"][-1]["content"] for r in new),
        "failures": fail,
    }
    json.dump(audit, open(AUDIT, "w"), indent=1)
    print(json.dumps({k: v for k, v in audit.items() if k not in ("tool_counts", "jaccard_top5")}, indent=1))
    if fail:
        print(f"\nABORT: {len(fail)} failure(s); {OUT} not written", file=sys.stderr)
        sys.exit(1)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {len(out)} records ({len(v3)} v3 + {len(new)} new) -> {OUT}")


if __name__ == "__main__":
    main()
