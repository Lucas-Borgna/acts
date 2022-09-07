import argparse
import pathlib
import acts
import acts.examples
from pathlib import Path

u = acts.UnitConstants

field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

from acts.examples.simulation import addPythia8


def generate_events(nevents: int, nthreads: int, pu: int, outdir: str):

    s = acts.examples.Sequencer(
        events=nevents, numThreads=nthreads, logLevel=acts.logging.INFO
    )

    outputDir = Path(outdir)
    outputDir.mkdir(exist_ok=True)
    outputDirCsv = Path(outdir + "/csv")
    outputDirCsv.mkdir(exist_ok=True)

    s = addPythia8(
        s,
        rnd=rnd,
        nhard=1,
        npileup=pu,
        hardProcess=["Top:qqbar2ttbar=on"],
        outputDirRoot=outputDir,
        outputDirCsv=outputDirCsv,
    )

    s.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ttbar events with Pythia8")
    parser.add_argument(
        "-n", "--nevents", type=int, default=100, help="Number of events to generate"
    )
    parser.add_argument(
        "--nthreads", type=int, default=1, help="Number of threads to use"
    )
    parser.add_argument(
        "--pu", type=int, default=200, help="Number of pileup events to generate"
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default="ttbar_pythia", help="Output directory"
    )

    args = parser.parse_args()

    generate_events(args.nevents, args.nthreads, args.pu, args.outdir)
