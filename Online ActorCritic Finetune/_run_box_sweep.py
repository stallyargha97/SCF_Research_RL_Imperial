import sys, subprocess, re, time
import nbformat

nb_path = sys.argv[1]
boxes = sys.argv[2:] if len(sys.argv) > 2 else ['Narrow', 'Mid', 'Wide']

for box in boxes:
    nb = nbformat.read(nb_path, as_version=4)
    patched = False
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        src = cell.source
        if re.search(r"^BOX = '[A-Za-z]+'", src, flags=re.MULTILINE):
            cell.source = re.sub(r"^BOX = '[A-Za-z]+'(.*)$", f"BOX = '{box}'\\1", src, count=1, flags=re.MULTILINE)
            patched = True
            break
    if not patched:
        print(f"[{nb_path}] ERROR: BOX line not found"); sys.exit(1)
    nbformat.write(nb, nb_path)
    print(f"[{nb_path}] BOX -> {box}, executing...", flush=True)

    t0 = time.time()
    result = subprocess.run(
        ['python', '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',
         '--ExecutePreprocessor.timeout=14400', nb_path],
        capture_output=True, text=True
    )
    dt = (time.time() - t0) / 60
    if result.returncode != 0:
        print(f"[{nb_path}] BOX={box} FAILED after {dt:.1f} min")
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        sys.exit(1)
    print(f"[{nb_path}] BOX={box} done in {dt:.1f} min", flush=True)

print(f"[{nb_path}] ALL BOXES DONE")
