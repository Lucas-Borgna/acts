from pathlib import Path
from typing import Optional, Union
import argparse
import acts
import acts.examples


u = acts.UnitConstants

from acts.examples.simulation import addFatras, addDigitization

from acts.examples.reconstruction import (
    addSeeding,
    addCKFTracks,
    addVertexFitting,
    VertexFinder,
    TruthSeedRanges,
    CKFPerformanceConfig,
)


def run_tracking(input_path: str, output_path: str, s=None):

    s = s or acts.examples.Sequencer(
        events=20, numThreads=4, logLevel=acts.logging.INFO
    )
    rnd = acts.examples.RandomNumbers(seed=42)

    # LOAD EVENTS (pre-generated)

    evGen = acts.examples.RootParticleReader(
        level=s.config.logLevel,
        particleCollection="particles_input",
        filePath=input_path,
        orderedEvents=False,
    )
    s.addReader(evGen)

    # SETUP DETECTOR
    outputDir = Path(output_path)
    outputDir.mkdir(exist_ok=True)
    outputDirCsv = Path(output_path + "/csv")
    outputDirCsv.mkdir(exist_ok=True)

    detector, trackingGeometry, _ = acts.examples.GenericDetector.create()
    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))

    s = addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
    )

    # SETUP DIGITIZATION
    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    digiConfigFile = (
        srcdir
        / "Examples/Algorithms/Digitization/share/default-smearing-config-generic.json"
    )

    s = addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        outputDirCsv=outputDirCsv,
        outputDirRoot=outputDir,
        rnd=rnd,
    )

    # Setup Seeding
    geometrySelection = (
        srcdir
        / "Examples/Algorithms/TrackFinding/share/geoSelection-genericDetector.json"
    )

    s = addSeeding(
        s,
        trackingGeometry,
        field,
        TruthSeedRanges(pt=(2.0 * u.GeV, None), eta=(-2.7, 2.7), nHits=(9, None)),
        geoSelectionConfigFile=geometrySelection,
        outputDirRoot=outputDir,
        initialVarInflation=[100, 100, 100, 100, 100, 100],
    )

    # Setup CKF Tracking

    s = addCKFTracks(
        s,
        trackingGeometry,
        field,
        CKFPerformanceConfig(ptMin=2000.0 * u.MeV, nMeasurementsMin=6),
        outputDirRoot=outputDir,
    )

    s.addAlgorithm(
        acts.examples.TrackSelector(
            level=acts.logging.INFO,
            inputTrackParameters="fittedTrackParameters",
            outputTrackParameters="trackparameters",
            outputTrackIndices="outputTrackIndices",
            removeNeutral=True,
            absEtaMax=2.5,
            loc0Max=4.0 * u.mm,
            ptMin=2000 * u.MeV,
        )
    )

    # Setup Vertex Fitting
    s = addVertexFitting(
        s,
        field,
        vertexFinder=VertexFinder.Iterative,
        outputDirRoot=outputDir,
        trajectories="trajectories",
    )

    s.run()


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Add detector layers to the geometry")
    parser.add_argument(
        "-i", "--input", type=str, default="ttbar_pythia/csv", help="Input directory"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="ttbar_pythia_sim/csv",
        help="Output directory",
    )
    args = parser.parse_args()
    run_tracking(input_path=args.input, output_path=args.output)
