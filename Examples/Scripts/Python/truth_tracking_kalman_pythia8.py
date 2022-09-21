#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Optional, Union

import acts
import acts.examples

u = acts.UnitConstants


def runTruthTrackingKalman(
    trackingGeometry: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    outputDir: Path,
    digiConfigFile: Path,
    decorators=[],
    directNavigation=False,
    reverseFilteringMomThreshold=0 * u.GeV,
    s: acts.examples.Sequencer = None,
    inputParticlePath: Optional[Path] = None,
):
    from acts.examples.simulation import (
        addParticleGun,
        EtaConfig,
        PhiConfig,
        ParticleConfig,
        addFatras,
        addDigitization,
    )
    from acts.examples.reconstruction import (
        addSeeding,
        SeedingAlgorithm,
        TruthSeedRanges,
        addKalmanTracks,
        addVertexFitting,
        VertexFinder,
    )

    s = s or acts.examples.Sequencer(
        events=100, numThreads=-1, logLevel=acts.logging.INFO
    )

    rnd = acts.examples.RandomNumbers()
    outputDir = Path(outputDir)

    if inputParticlePath is None:
        addParticleGun(
            s,
            EtaConfig(-2.0, 2.0),
            ParticleConfig(2, acts.PdgParticle.eMuon, False),
            multiplicity=1,
            rnd=rnd,
            outputDirRoot=outputDir,
        )
    else:
        acts.logging.getLogger("Truth tracking example").info(
            "Reading particles from %s", inputParticlePath.resolve()
        )
        assert inputParticlePath.exists()
        s.addReader(
            acts.examples.RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                particleCollection="particles_input",
                orderedEvents=False,
            )
        )

    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
    )

    addSeeding(
        s,
        trackingGeometry,
        field,
        seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        rnd=rnd,
        truthSeedRanges=TruthSeedRanges(
            pt=(500 * u.MeV, None),
            nHits=(9, None),
        ),
    )

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

    s.addWriter(
        acts.examples.TrackFinderPerformanceWriter(
            level=acts.logging.INFO,
            inputProtoTracks="sortedprototracks" if directNavigation else "prototracks",
            inputParticles="truth_seeds_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "performance_track_finder.root"),
        )
    )

    s.addWriter(
        acts.examples.TrackFitterPerformanceWriter(
            level=acts.logging.INFO,
            inputTrajectories="trajectories",
            inputParticles="truth_seeds_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "performance_track_fitter.root"),
        )
    )
    
    s.addAlgorithm(
        acts.examples.TrackSelector(
            level=acts.logging.INFO,
            inputTrackParameters="fittedTrackParameters",
            outputTrackParameters="trackparameters",
            outputTrackIndices="outputTrackIndices",
            removeNeutral=True,
            absEtaMax=2.5,
            loc0Max=4.0 * u.mm, # rho max
            ptMin=500 * u.MeV,
        )
    )
    addVertexFitting(
        s,
        field,
        vertexFinder=VertexFinder.Iterative
        outputDirRoot=outputDir,
        trajectories="trajectories",
    )

    return s


if "__main__" == __name__:

    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    # detector, trackingGeometry, _ = getOpenDataDetector()
    detector, trackingGeometry, decorators = acts.examples.GenericDetector.create()

    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
    outputDir= Path(str(Path.cwd()) + '/truth_tracking_kalman/') 
    inputParticlePath = Path(str(Path.cwd()) + "/data/gen/ttbar200-vxgen/pythia8_particles.root")

    runTruthTrackingKalman(
        trackingGeometry,
        field,
        digiConfigFile=srcdir
        / "Examples/Algorithms/Digitization/share/default-smearing-config-generic.json",
        # "thirdparty/OpenDataDetector/config/odd-digi-smearing-config.json",
        outputDir=outputDir,
        inputParticlePath=inputParticlePath
    ).run()
