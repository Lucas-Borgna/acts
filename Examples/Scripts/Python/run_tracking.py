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
    addKalmanTracks,
    addVertexFitting,
    VertexFinder,
    TruthSeedRanges,
    CKFPerformanceConfig,
)


def run_tracking(input_path: str, output_path: str, truth_tracking: bool, s=None):

    s = s or acts.examples.Sequencer(events=1, numThreads=1, logLevel=acts.logging.INFO)
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
    # Tracking setup
    if truth_tracking:
        # Setup Truth Tracking
        directNavigation = False
        reverseFilteringMomThreshold = 0 * u.GeV
        addKalmanTracks(
            s, trackingGeometry, field, directNavigation, reverseFilteringMomThreshold
        )

        # Output
        s.addWriter(
            acts.examples.RootTrajectoryStatesWriter(
                level=acts.logging.INFO,
                inputTrajectories="trajectories",
                inputParticles="truth_seeds_selected",
                inputSimHits="simhits",
                inputMeasurementParticlesMap="measurement_particles_map",
                inputMeasurementSimHitsMap="measurement_simhits_map",
                filePath=str(outputDir / "trackstates_fitter.root"),
            )
        )

        s.addWriter(
            acts.examples.RootTrajectorySummaryWriter(
                level=acts.logging.INFO,
                inputTrajectories="trajectories",
                inputParticles="truth_seeds_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
                filePath=str(outputDir / "tracksummary_fitter.root"),
            )
        )

        # s.addWriter(
        #     acts.examples.TrackFinderPerformanceWriter(
        #         level=acts.logging.INFO,
        #         inputProtoTracks="sortedprototracks" if directNavigation else "prototracks",
        #         inputParticles="truth_seeds_selected",
        #         inputMeasurementParticlesMap="measurement_particles_map",
        #         filePath=str(outputDir / "performance_track_finder.root"),
        #     )
        # )
        s.addWriter(
            acts.examples.TrackFitterPerformanceWriter(
                level=acts.logging.INFO,
                inputTrajectories="trajectories",
                inputParticles="truth_seeds_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
                filePath=str(outputDir / "performance_track_fitter.root"),
            )
        )

    else:
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
    parser.add_argument(
        "-t",
        "--truth-tracking",
        default=False,
        help="Truth tracking",
        action="store_true",
    )
    args = parser.parse_args()
    run_tracking(
        input_path=args.input,
        output_path=args.output,
        truth_tracking=args.truth_tracking,
    )
