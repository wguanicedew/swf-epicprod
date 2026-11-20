#!/usr/bin/env python3
"""Clone a particle-gun task's configuration into a new Event Service task.

The new task is a whole production task, not a PCS trial: a production
sample of the configuration (pcs/production_samples.py compose) with its own
sample name (``cES000001`` onward), random seeds disjoint from the source
production and from every ES sample on any queue, a generation plan in BNL
Rucio, and RECO in the production namespace with normal retention.

What is cloned from SOURCE (any task of the configuration: the production
task itself, one of its trials such as ``….3to50deg.trial2``, or one of its
ES samples such as ``….3to50deg.gOP000063``):
  - the configuration it is a variant of: the tags and the gun sample, read
    from the task's production-sample provenance, its trial subject, or the
    task itself;
  - its bound ProdConfig and its overrides, less identity, placement and
    feeder markers;
  - its Event Service shape (``overrides['event_service']``), less the
    per-sample identity (events, input dataset, row), or the shape of
    --es-from when SOURCE carries none (a trial cannot carry one: the ES
    producer refuses a trial spec).
--events, --events-per-job, --set KEY=JSON and --unset KEY adjust the shape.

Default is a read-only plan. --apply composes the task, placed on --site,
and registers its generation plan. --submit (with --apply) then makes the
same submit request as the PCS Submit button (services.prodtask_submit_request),
which refuses Event Service work on a queue outside the whole-node sites
(swf_epicprod/front.py event_service_misplaced): BNL_OSG_PanDA_1's pilot
failed every ES job on 2026-10-29 (pilot error 1310). The plan prints that
verdict before anything is written.

--submit-direct submits to PanDA from this process instead of through the
prod-ops agent: the same PCS bookkeeping as the Submit button (release tags
bound, a PandaTasks attempt allocated with its placement and submitter), then
the deployed monitor's scripts/submit-evgen-task.py doer is run here with
production.env, the operator's SWF_API_TOKEN from ~/.env and the internal
monitor URL (as submit-es-canary.py runs it). The doer records the jediTaskID
(or the failure) on the attempt. The Event Service placement verdict still
refuses unless --ignore-placement names the override. With --apply it submits
the new clone; without --apply SOURCE must be an unsubmitted clone made
earlier by this script, which is then submitted as it stands.

--restore puts back the Event Service setup of a clone that lost it: an
Unplace of a task not yet in PanDA drops ``overrides['event_service']``
(pcs/services.py prodtask_unplace), and no whole-node queue rebuilds a
particle-gun shape. The setup is the one recorded on the clone at --apply;
a clone made before that record is rebuilt from its --es-from task with the
--events-per-job, --set and --unset given now (the --apply ones: they were
not recorded), and refused unless its seed stem matches the recorded one, so
it names the generation plan already registered. --site places it again;
--submit-direct then submits it.

Run in the monitor Django environment with the testbed venv and ~/.env, as
the production-ops user (the plan upload needs the service proxy).
--production-env and --env-file PATH load KEY=VALUE files into this process
before Django starts (production.env cannot be sourced: its values carry shell
metacharacters); --submit-direct's attempt-name lookup needs the PANDA_DB_*
settings production.env holds. Values are never printed, only key counts.

    clone-es-task.py <source> --by wguan --reason TEXT [--site BNL_OSG_PanDA_1] \\
        [--es-from <ES task>] [--events N] [--set slots=1] \\
        [--apply [--submit | --submit-direct [--ignore-placement]]]
    clone-es-task.py <clone> --by wguan --reason TEXT --submit-direct
    clone-es-task.py <clone> --by wguan --reason TEXT --restore [--events-per-job N] \\
        [--site QUEUE] [--submit-direct [--ignore-placement]]
    clone-es-task.py <clone> --by wguan --reason TEXT --production-env \\
        --fail-pending [TRY ...] [--min-age MINUTES] [--apply]

--fail-pending marks the clone's attempts that show pending (no jediTaskID,
no recorded failure) as submit_failed: those older than --min-age whose name
PanDA does not hold, checked by the same jedi_tasks lookup that allocates
names (so it needs the PANDA_DB_* settings, --production-env).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

PRODUCTION_ENV = Path('/opt/swf-monitor/config/env/production.env')


def key_values(path):
    """A KEY=VALUE env file read as data, never sourced (submit-es-canary.py):
    production.env's values carry shell metacharacters."""
    values = {}
    for raw in Path(path).read_text().splitlines():
        raw = raw.strip().removeprefix('export ')
        if raw and not raw.startswith('#') and '=' in raw:
            key, value = raw.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_files(argv):
    """The --production-env / --env-file files, in the order given; read
    before Django starts, as its settings (the PanDA DB among them) come
    from the environment."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--production-env', action='store_true')
    pre.add_argument('--env-file', action='append', default=[])
    known, _ = pre.parse_known_args(argv)
    return ([PRODUCTION_ENV] if known.production_env else []) + \
        [Path(p).expanduser() for p in known.env_file]


def load_env(files):
    """Each file's keys into this process's environment; a later file wins
    over an earlier one, and both over the shell's. Returns the keys set."""
    loaded = []
    for path in files:
        try:
            values = key_values(path)
        except OSError as e:
            raise SystemExit(f'--env-file {path}: {e.strerror}')
        os.environ.update(values)
        loaded.append((str(path), sorted(values)))
    return loaded


LOADED_ENV = load_env(env_files(sys.argv[1:]))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'swf_monitor_project.settings')

import django  # noqa: E402
django.setup()

from django.db import transaction  # noqa: E402

from pcs.commands import (build_evgen_task_params, internal_evgen_sample,  # noqa: E402
                          is_particle_gun, prodtask_events_per_job)
from pcs.models import ProdTask  # noqa: E402
from pcs.name_tokens import trial_subject_name  # noqa: E402
from pcs.production_samples import compose  # noqa: E402
from swf_epicprod.front import event_service_misplaced  # noqa: E402
from swf_epicprod.google_production import (overlaps, plan_names,  # noqa: E402
                                            register_plan, seed_intervals)
from swf_epicprod.production_es import definition  # noqa: E402
from swf_epicprod.queues import whole_node  # noqa: E402

MARKER = 'es_clone'
# Per-sample identity of an ES setup: recomputed for the new sample.
ES_IDENTITY = ('events', 'input_events', 'input_dataset', 'row', 'files')
# Overrides that name a task's identity, placement or the feeder that made it,
# or that compose sets itself: never carried to the clone.
NOT_CLONED = ('event_service', 'production_sample_prefix', 'panda_site', 'queued',
              'task_priority', 'google_production', 'perlmutter_production',
              'quiet_end', 'es_trial', MARKER,
              'copy_reco', 'copy_full', 'copy_log', 'use_rucio')
SPEC_KEYS = ('taskName', 'containerImage', 'inputDataset', 'nEventsPerJob', 'nJobs', 'nEvents',
             'nEventsPerWorker', 'fineGrainedRangeEvents', 'nCore', 'memory', 'walltimeHours',
             'site', 'exec')

# The deployed monitor whose credentialed doer submits (submit-es-canary.py).
DEPLOYED = Path('/opt/swf-monitor/current')
# The internal face: the external one refuses the record write.
MONITOR_URL = 'https://pandaserver02.sdcc.bnl.gov/swf-monitor'
DIRECT_SOURCE = 'clone_es_task_direct'
LOG_DIR = Path.home() / 'clone-es-task-logs'


def configuration_of(task):
    """The production task whose configuration ``task`` is a variant of."""
    ds = task.dataset
    md = (ds.metadata or {}) if ds is not None else {}
    production = md.get('production_sample') or {}
    if production.get('source'):
        name = production['source']
    elif md.get('trial'):
        name = trial_subject_name(task.composed_name)
    else:
        return task
    try:
        return ProdTask.objects.get(name=name)
    except ProdTask.DoesNotExist:
        raise SystemExit(f'{task.composed_name}: its configuration task {name} is not in PCS')


def es_shape(task, args):
    """The source's ES shape without its sample identity, with the edits."""
    es = (task.overrides or {}).get('event_service')
    if not es:
        raise SystemExit(f'{task.composed_name} carries no Event Service setup; '
                         'name an ES task to take the shape from with --es-from')
    if es.get('input') == 'evgen':
        raise SystemExit(f'{task.composed_name} is EVGEN-input ES; this clones particle-gun ES')
    shape = {k: v for k, v in es.items() if k not in ES_IDENTITY}
    for item in args.unset:
        shape.pop(item, None)
    for item in args.set:
        key, sep, value = item.partition('=')
        if not sep:
            raise SystemExit(f'--set {item!r}: expected KEY=VALUE')
        try:
            shape[key] = json.loads(value)
        except json.JSONDecodeError:
            shape[key] = value  # a bare string, e.g. slots=auto
    events = args.events or int(es['events'])
    if args.events_per_job:
        shape['events_per_job'] = args.events_per_job
    shape.update(events=events, input_events=events)
    return shape


def reserved_seeds(config):
    """Every seed interval taken: the configuration's production chunks and
    every gun ES sample's whole budget (as google_production.reserve reads)."""
    path, original_stem = internal_evgen_sample(config)
    chunks = -(-int(config.dataset.expected_events) // prodtask_events_per_job(config))
    reserved = seed_intervals(original_stem, chunks)
    samples = set()
    for task in ProdTask.objects.filter(overrides__has_key='event_service'):
        es = task.overrides['event_service']
        if es.get('input') == 'evgen':
            continue  # EVGEN-input ES takes no gun seeds
        stem = es['row'].split(',')[0].rsplit('/', 1)[1]
        reserved.extend(seed_intervals(stem, int(es['events'])))
        samples.add(task.overrides.get('production_sample_prefix'))
    return path, original_stem, reserved, samples


def allocate(config, events, prefix, wanted=''):
    """The first free ``<prefix>NNNNNN`` sample name, or ``wanted`` when free."""
    path, original_stem, reserved, samples = reserved_seeds(config)
    names = [wanted] if wanted else (f'{prefix}{i:06d}' for i in range(1, 1000000))
    for sample in names:
        stem = original_stem + '.' + sample
        seeds = seed_intervals(stem, events)
        if (sample not in samples and sum(b - a + 1 for a, b in seeds) == events
                and not overlaps(seeds, reserved)):
            return path, sample, stem
        if wanted:
            raise SystemExit(f'{wanted}: name taken or its seeds overlap existing production')
    raise SystemExit(f'no seed-disjoint {prefix} sample name remains')


def identify(es, config, path, sample, stem):
    events = es['events']
    es.update(input_dataset=plan_names(config, sample)[1],
              row=f'{path}.{sample}/{stem},steer,{events},0000')
    return definition(es)  # refuses a malformed shape before anything is written


def doer_environment(owner):
    """The deployed doer's environment: production.env, the operator's DRF
    token for the record write, the internal monitor URL."""
    token = key_values(Path.home() / '.env').get('SWF_API_TOKEN')
    if not token:
        raise SystemExit('ERROR: SWF_API_TOKEN is not in ~/.env')
    env = dict(os.environ, **key_values(PRODUCTION_ENV))
    env.update(SWFMON_TOKEN=token, SWF_TASK_OWNER=owner, SWF_MONITOR_URL=MONITOR_URL)
    return env


def prepare_direct(task, *, by, reason, ignore_placement):
    """PCS's side of a submission, as prodtask_submit_request does it, less
    the agent queue: the gates, the release tags bound, the attempt
    allocated with its submitter and placement. Returns the PandaTasks row."""
    from pcs.commands import pinned_site
    from pcs.services import (ServiceError, _clear_queued, _placement_record,
                              prodtask_allocate_panda_tasks, prodtask_bind_release_tags,
                              refuse_closed_queue)
    if task.panda_task_id is not None:
        raise ServiceError(f'Already submitted as jediTaskID {task.panda_task_id}.')
    site = pinned_site(task)
    if not site:
        raise ServiceError(f'{task.composed_name} is placed on no queue.')
    if not (task.overrides or {}).get('event_service'):
        raise ServiceError(f'{task.composed_name} carries no Event Service setup.')
    misplaced = event_service_misplaced(site, task)
    if misplaced and not ignore_placement:
        raise ServiceError(f'{misplaced} (--ignore-placement overrides, on your authority)')
    refuse_closed_queue(site, task=task)
    prodtask_bind_release_tags(task, changed_by=by)
    task.refresh_from_db()
    attempt = prodtask_allocate_panda_tasks(task=task, source=DIRECT_SOURCE)
    meta = dict(attempt.metadata or {}, submitted_by=by, direct_submission=True,
                placement=_placement_record(site, {'reason': reason}, by))
    if misplaced:
        meta['placement_overridden'] = misplaced
    attempt.metadata = meta
    attempt.site = site
    attempt.save(update_fields=['metadata', 'site', 'updated_at'])
    _clear_queued(task)
    return attempt


# The doer's settings an operator outside the production-ops account points
# elsewhere: option -> (what it is, how the doer takes it).
DOER_PATHS = {
    'jlab_proxy': 'the JLab eicprod x509 proxy shipped in the sandbox (--proxy, '
                  'EVGEN_X509_PROXY)',
    'login_proxy': 'the robot proxy (longproxy-for-rucio) that signs in to the PanDA server '
                   'for the JLab Rucio tokens (EPICPROD_PANDA_LOGIN_PROXY, X509_USER_PROXY)',
    'tmp_dir': 'the scratch root the sandbox is assembled under (SWF_TMP_DIR)',
    'pclient_setup': 'the panda-client setup the submission kernel sources, holding the '
                     'cached OIDC token (--pclient-setup, default ~/pclient/run/setup.sh)',
    'reporting_env': "the job report's S3 settings (JOB_REPORTING_ENV, default "
                     '~/.epic-job-reporter.env)',
}


def doer_options(args):
    """The DOER_PATHS options given, as absolute paths."""
    return {k: os.path.abspath(os.path.expanduser(getattr(args, k)))
            for k in DOER_PATHS if getattr(args, k, '')}


def doer_problem(doer):
    """Why the doer options cannot work, or ''; read before an attempt is
    allocated, so a refusal leaves nothing to clean up."""
    for key in ('jlab_proxy', 'login_proxy', 'pclient_setup', 'reporting_env'):
        path = doer.get(key)
        if path and not os.access(path, os.R_OK):
            return f'--{key.replace("_", "-")} {path} is not readable'
    tmp = doer.get('tmp_dir')
    if tmp:
        root = Path(tmp, 'submit-evgen')
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return f'--tmp-dir {tmp}: {e.strerror}'
        if not os.access(root, os.W_OK | os.X_OK):
            return f'--tmp-dir {root} is not writable'
    return ''


def run_doer(task, attempt, *, owner, timeout, log, doer=None):
    """Run the deployed submit-evgen-task.py for the allocated attempt.
    Returns (exit code, jediTaskID or None). The doer records the outcome.
    ``doer`` replaces production.env's settings (DOER_PATHS)."""
    doer = doer or {}
    cmd = [str(DEPLOYED / '.venv/bin/python'), str(DEPLOYED / 'scripts/submit-evgen-task.py'),
           '--task-name', task.name, '--panda-tasks-id', str(attempt.pk), '--owner', owner]
    env = doer_environment(owner)
    if doer.get('jlab_proxy'):
        cmd += ['--proxy', doer['jlab_proxy']]
        env['EVGEN_X509_PROXY'] = doer['jlab_proxy']
    if doer.get('login_proxy'):
        env.update(EPICPROD_PANDA_LOGIN_PROXY=doer['login_proxy'],
                   X509_USER_PROXY=doer['login_proxy'])
    if doer.get('tmp_dir'):
        env['SWF_TMP_DIR'] = doer['tmp_dir']
    if doer.get('pclient_setup'):
        cmd += ['--pclient-setup', doer['pclient_setup']]
    if doer.get('reporting_env'):
        env['JOB_REPORTING_ENV'] = doer['reporting_env']
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        log.write_text(f'{" ".join(cmd)}\nTIMEOUT after {timeout}s\n{e.stderr or ""}')
        return 4, None
    log.write_text(f'{" ".join(cmd)}\n--- stderr\n{p.stderr}\n--- stdout\n{p.stdout}'
                   f'\n--- exit {p.returncode}\n')
    out = (p.stdout or '').strip()
    return p.returncode, (int(out) if out.isdigit() else None)


def submit_direct(task, *, by, reason, ignore_placement=False, timeout=360, log=None,
                  doer=None):
    """Submit ``task`` to PanDA from this process (no prod-ops agent).
    Returns what happened, for the plan's ``submitted``."""
    from pcs.services import ServiceError
    doer = doer or {}
    problem = doer_problem(doer)
    if problem:
        return {'outcome': 'refused', 'reason': problem}
    try:
        attempt = prepare_direct(task, by=by, reason=reason, ignore_placement=ignore_placement)
    except ServiceError as e:
        reason = str(e)
        if e.__cause__ is not None:  # e.g. the PanDA name lookup's own error
            reason += f' ({type(e.__cause__).__name__}: {e.__cause__})'
        return {'outcome': 'refused', 'reason': reason}
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    log = log or LOG_DIR / f'{task.composed_name}.{stamp}.log'
    owner = task.created_by or by
    try:
        rc, jedi = run_doer(task, attempt, owner=owner, timeout=timeout, log=log, doer=doer)
    except (OSError, SystemExit) as e:
        attempt.delete()  # nothing reached PanDA: free the attempt name
        return {'outcome': 'not run', 'reason': str(e), 'panda_task_name': attempt.task_name}
    outcome = {0: 'submitted and recorded',
               7: 'submitted, NOT recorded: re-record the jediTaskID (record-submission)',
               8: 'output datasets not created; nothing submitted'}.get(
                   rc, 'failed (the doer recorded the failure on the attempt)')
    if rc not in (0, 7):
        # The doer records a failure through the monitor API, and an early
        # exit or an unwritable record leaves the attempt showing pending.
        attempt.refresh_from_db()
        if attempt.jedi_task_id is None and attempt.status_snapshot != 'submit_failed':
            from pcs.services import prodtask_record_submission_failure
            prodtask_record_submission_failure(
                task=task, panda_tasks_id=attempt.pk,
                reason=f'submit-evgen-task.py exit {rc}; see {log}')
            outcome = f'failed, exit {rc} (recorded on the attempt here)'
    return {'outcome': outcome, 'exit': rc, 'jedi_task_id': jedi,
            'panda_task_name': attempt.task_name, 'panda_tasks_id': attempt.pk,
            'log': str(log)}


def clone_of(name):
    """A task this script made, or SystemExit."""
    task = ProdTask.objects.get(name=name)
    if MARKER not in (task.overrides or {}):
        raise SystemExit(f'{task.composed_name} was not made by clone-es-task.py; '
                         'clone it with --apply')
    return task


def restored_setup(task, args):
    """The clone's Event Service setup as --apply made it."""
    entry = task.overrides[MARKER]
    if entry.get('event_service'):
        return dict(entry['event_service'])
    # A clone made before the setup was recorded: rebuilt, then checked
    # against the seed stem its registered plan was made from.
    es_source = ProdTask.objects.get(name=entry['es_from'])
    config = configuration_of(task)
    args.events = int(entry['events'])
    es = es_shape(es_source, args)
    path, original_stem = internal_evgen_sample(config)
    stem = original_stem + '.' + entry['sample']
    if stem != entry['seed_stem']:
        raise SystemExit(f'rebuilt seed stem {stem} is not the recorded {entry["seed_stem"]}')
    return identify(es, config, path, entry['sample'], stem)


def restore(args):
    """--restore: put back a clone's Event Service setup, place it, submit it."""
    from pcs.services import ServiceError, prodtask_place
    task = clone_of(args.source)
    result = {'task': task.composed_name}
    if (task.overrides or {}).get('event_service'):
        result['restored'] = 'already carries its Event Service setup'
    else:
        if task.panda_task_id is not None:
            raise SystemExit(f'{task.composed_name} is in PanDA as {task.panda_task_id}: '
                             'its jobs already have their shape')
        es = restored_setup(task, args)
        with transaction.atomic():
            locked = ProdTask.objects.select_for_update().get(pk=task.pk)
            overrides = dict(locked.overrides or {}, event_service=es)
            overrides[MARKER] = dict(overrides[MARKER], event_service=es)
            locked.overrides = overrides
            locked.save(update_fields=['overrides', 'updated_at'])
        task.refresh_from_db()
        result['restored'] = es
    if args.site:
        try:
            prodtask_place(task, args.site, changed_by=args.by,
                           placement={'reason': args.reason})
            task.refresh_from_db()
        except ServiceError as e:
            raise SystemExit(f'placing on {args.site}: {e.detail}')
    site = (task.overrides or {}).get('panda_site') or ''
    verdict = event_service_misplaced(site, task) if site else 'placed on no queue'
    result.update(site=site or None, submission_verdict=verdict or 'accepted')
    if args.submit_direct:
        result['submitted'] = submit_direct(task, by=args.by, reason=args.reason,
                                            ignore_placement=args.ignore_placement,
                                            timeout=args.submit_timeout, doer=doer_options(args))
    print(json.dumps(result, indent=2, default=str))
    return 0 if not args.submit_direct or result['submitted'].get('exit') == 0 else 1


def submit_existing(args):
    """--submit-direct without --apply: submit a clone this script made."""
    task = clone_of(args.source)
    result = submit_direct(task, by=args.by, reason=args.reason,
                           ignore_placement=args.ignore_placement, timeout=args.submit_timeout,
                           doer=doer_options(args))
    print(json.dumps({'task': task.composed_name, 'site': (task.overrides or {}).get('panda_site'),
                      'submitted': result}, indent=2, default=str))
    return 0 if result.get('exit') == 0 else 1


def pending_attempts(task, tries=(), min_age_minutes=30):
    """The clone's attempts that show pending and never reached PanDA: no
    jediTaskID and no recorded failure. Each is returned with why it is or
    is not to be marked failed. ``tries`` limits them to those try numbers;
    an attempt younger than ``min_age_minutes`` is left alone, as a
    submission may still be running; one whose name PanDA holds is left
    alone, as it reached PanDA and wants its jediTaskID recorded instead."""
    from django.utils import timezone
    from pcs.services import ServiceError, _panda_task_name_exists
    rows = []
    now = timezone.now()
    for row in task.panda_tasks.order_by('try_number'):
        if row.jedi_task_id is not None or row.status_snapshot:
            continue
        if tries and row.try_number not in tries:
            continue
        age = (now - row.created_at).total_seconds() / 60
        entry = {'panda_tasks_id': row.pk, 'try': row.try_number, 'name': row.task_name,
                 'created_at': row.created_at, 'age_minutes': round(age)}
        if age < min_age_minutes:
            entry['skip'] = f'younger than {min_age_minutes} minutes: may still be submitting'
        else:
            try:
                if _panda_task_name_exists(row.task_name):
                    entry['skip'] = ('PanDA holds this name: it reached PanDA; record its '
                                     'jediTaskID instead')
            except ServiceError as e:
                cause = f' ({e.__cause__})' if e.__cause__ is not None else ''
                entry['skip'] = f'{e}{cause}: run with --production-env'
        rows.append(entry)
    return rows


def fail_pending(args):
    """--fail-pending: mark a clone's stuck attempts submit_failed. A plan
    unless --apply."""
    from pcs.services import prodtask_record_submission_failure
    task = clone_of(args.source)
    rows = pending_attempts(task, tries=set(args.fail_pending),
                            min_age_minutes=args.min_age)
    for entry in rows:
        if 'skip' in entry:
            continue
        if args.apply:
            prodtask_record_submission_failure(
                task=task, panda_tasks_id=entry['panda_tasks_id'],
                reason=f'marked failed by {args.by} (clone-es-task.py --fail-pending): '
                       f'{args.reason}; it never reached PanDA')
            entry['marked'] = 'submit_failed'
        else:
            entry['marked'] = 'would mark submit_failed (--apply to write)'
    print(json.dumps({'task': task.composed_name, 'attempts': rows}, indent=2, default=str))
    if not args.apply:
        print('plan only: --apply marks the attempts', file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('source', help='composed name of the task whose configuration is cloned')
    ap.add_argument('--by', required=True, help='operator creating the task (records carry it)')
    ap.add_argument('--reason', required=True,
                    help='why, and whose approval: recorded as the authorization')
    ap.add_argument('--site', default='',
                    help='queue to place on (default BNL_OSG_PanDA_1 for a new clone; '
                         'with --restore, left as it is)')
    ap.add_argument('--es-from', default='', help='task whose ES shape is cloned (default: source)')
    ap.add_argument('--prefix', default='cES', help='sample name prefix (default cES)')
    ap.add_argument('--sample', default='', help='explicit sample name (default: next free)')
    ap.add_argument('--events', type=int, default=0, help="default: the ES source's events")
    ap.add_argument('--events-per-job', type=int, default=0)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=JSON',
                    help='set an ES shape key, e.g. --set slots=1 --set pilot_cores=1')
    ap.add_argument('--unset', action='append', default=[], metavar='KEY',
                    help='drop an ES shape key, e.g. --unset resident_sim')
    ap.add_argument('--proxy', default='', help='x509 proxy for the plan upload '
                    '(default: the GKE settings proxy, as evgen-es-submit.py uses)')
    ap.add_argument('--apply', action='store_true', help='create the task; default plans only')
    ap.add_argument('--submit', action='store_true',
                    help="with --apply: make PCS's submit request for the new task")
    ap.add_argument('--submit-direct', action='store_true',
                    help='submit to PanDA from here with the deployed doer, not via the '
                         'prod-ops agent; without --apply, SOURCE is an existing clone')
    ap.add_argument('--ignore-placement', action='store_true',
                    help='with --submit-direct: submit despite the Event Service placement '
                         'verdict (recorded on the attempt)')
    for key, what in DOER_PATHS.items():
        ap.add_argument('--' + key.replace('_', '-'), default='', metavar='PATH',
                        help=f'with --submit-direct: {what}, in place of production.env\'s '
                             'or the production-ops account\'s')
    ap.add_argument('--submit-timeout', type=int, default=360,
                    help='seconds the direct doer may take (default 360)')
    ap.add_argument('--fail-pending', nargs='*', type=int, default=None, metavar='TRY',
                    help="mark the clone's attempts that show pending and never reached PanDA "
                         'as submit_failed (all of them, or these try numbers; base is 1); '
                         'a plan unless --apply')
    ap.add_argument('--min-age', type=int, default=30, metavar='MINUTES',
                    help='with --fail-pending: leave attempts younger than this alone '
                         '(default 30)')
    ap.add_argument('--restore', action='store_true',
                    help="put back the Event Service setup of a clone that lost it (Unplace)")
    ap.add_argument('--production-env', action='store_true',
                    help=f'load {PRODUCTION_ENV} (read as KEY=VALUE data) before Django '
                         'starts: the PanDA DB settings the attempt-name lookup needs')
    ap.add_argument('--env-file', action='append', default=[], metavar='PATH',
                    help='load another KEY=VALUE file, e.g. ~/.env (repeatable; '
                         'applied after --production-env, a later file wins)')
    args = ap.parse_args()
    for path, keys in LOADED_ENV:
        print(f'env: {path}: {len(keys)} keys', file=sys.stderr)
    if args.fail_pending is not None:
        if args.restore or args.submit or args.submit_direct:
            ap.error('--fail-pending stands alone (with --apply to write)')
        return fail_pending(args)
    if args.restore:
        if args.apply or args.submit:
            ap.error('--restore acts on an existing clone: no --apply or --submit')
        if args.ignore_placement and not args.submit_direct:
            ap.error('--ignore-placement is for --submit-direct')
        return restore(args)
    args.site = args.site or 'BNL_OSG_PanDA_1'
    if args.submit and not args.apply:
        ap.error('--submit needs --apply')
    if args.submit and args.submit_direct:
        ap.error('--submit and --submit-direct are alternatives')
    if args.ignore_placement and not args.submit_direct:
        ap.error('--ignore-placement is for --submit-direct')
    if args.submit_direct and not args.apply:
        return submit_existing(args)

    source = ProdTask.objects.get(name=args.source)
    es_source = ProdTask.objects.get(name=args.es_from) if args.es_from else source
    config = configuration_of(source)
    if not is_particle_gun(config):
        raise SystemExit(f'{config.composed_name} is not an internally generated particle gun')
    if config.dataset.background_tag_id:
        raise SystemExit('background-mixed samples need a separate ES validation')
    if (config.overrides or {}).get('event_service') or (config.dataset.metadata or {}).get('trial'):
        raise SystemExit(f'{config.composed_name}: resolved configuration is itself a sample or trial')
    if configuration_of(es_source).pk != config.pk:
        raise SystemExit(f'{es_source.composed_name} is not of the configuration '
                         f'{config.composed_name}')

    es = es_shape(es_source, args)
    extra = {k: v for k, v in (source.overrides or {}).items() if k not in NOT_CLONED}
    prod_config = source.prod_config or config.prod_config
    path, sample, stem = allocate(config, es['events'], args.prefix, args.sample)
    es = identify(es, config, path, sample, stem)
    probe = ProdTask(name='(clone)', dataset=config.dataset, prod_config=prod_config,
                     overrides=dict(extra, event_service=es, panda_site=args.site))
    verdict = event_service_misplaced(args.site, probe)
    plan = {'source': source.composed_name, 'configuration': config.composed_name,
            'es_shape_from': es_source.composed_name,
            'prod_config': getattr(prod_config, 'name', None), 'sample': sample,
            'site': args.site, 'whole_node_site': whole_node(args.site),
            'event_service': es, 'cloned_overrides': extra,
            'submission_verdict': verdict or 'accepted'}
    if not args.apply:
        print(json.dumps(plan, indent=2, default=str))
        print('plan only: --apply creates the task and registers its generation plan',
              file=sys.stderr)
        return 0

    with transaction.atomic():
        # The configuration's row lock serializes this with the feeders'
        # reservations; the name is allocated afresh under it.
        locked = ProdTask.objects.select_for_update().get(pk=config.pk)
        path, sample, stem = allocate(locked, es['events'], args.prefix, args.sample)
        es = identify(es, locked, path, sample, stem)
        plan['sample'] = sample
        # compose records its standard authorization; this clone's own
        # authority (--reason) is kept on the MARKER override below.
        task = compose(locked, sample=sample, event_service=es, site=args.site,
                       created_by=args.by)
        task.overrides.update(extra)
        task.overrides[MARKER] = {'authority': args.reason, 'by': args.by, 'sample': sample,
                                  'events': es['events'], 'seed_stem': stem,
                                  'source': source.composed_name,
                                  'es_from': es_source.composed_name,
                                  # what --restore puts back after an Unplace
                                  'event_service': es}
        fields = ['overrides']
        if prod_config is not None and task.prod_config_id != prod_config.pk:
            task.prod_config = prod_config
            fields.append('prod_config')
        task.save(update_fields=fields)
    proxy = args.proxy
    if not proxy:
        from swf_epicprod.gke_pilots import settings as gke_settings
        proxy = gke_settings()['proxy']
    register_plan(task, proxy, marker=MARKER)
    spec = build_evgen_task_params(task)
    verdict = event_service_misplaced(args.site, task)
    plan['submission_verdict'] = verdict or 'accepted'
    plan.update(task=task.composed_name, spec={k: spec.get(k) for k in SPEC_KEYS})
    if args.submit:
        from pcs.services import ServiceError, prodtask_submit_request
        try:
            prodtask_submit_request(task=task, changed_by=args.by, source='clone_es_task')
            plan['submitted'] = 'requested: the prod-ops agent records the jediTaskID'
        except ServiceError as e:
            plan['submitted'] = f'refused: {e}'
    elif args.submit_direct:
        plan['submitted'] = submit_direct(task, by=args.by, reason=args.reason,
                                          ignore_placement=args.ignore_placement,
                                          timeout=args.submit_timeout, doer=doer_options(args))
    print(json.dumps(plan, indent=2, default=str))
    if verdict and not (args.submit_direct and args.ignore_placement):
        print(f'NOT SUBMITTABLE as composed: {verdict}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
