#!/usr/bin/env nextflow 
nextflow.enable.dsl=2

//params.fasta = WorkflowMain.getGenomeAttribute(params, 'fasta')

include { make_bintable } from './modules/local/make_bintable'
include { make_track_db } from './modules/local/make_track_db'
include { eigdecomp } from './modules/local/eigdecomp'
include { plot_spectrum} from './modules/local/plot_spectrum'
// include { make_multivec } from './modules/local/make_multivec'
include { clustering } from './modules/local/clustering'
include { heatmap } from './modules/local/heatmap'
include { scatters } from './modules/local/scatters'

workflow {
    make_bintable(params.config, params.genome)
    make_track_db(params.config, make_bintable.out.bins, params.meta) 
    eigdecomp(params.config, make_bintable.out.bins, params.mcool)
    plot_spectrum(params.config, eigdecomp.out.eigvals)
    // make_multivec(params.config, eigdecomp.out.eigvals, eigdecomp.out.eigvecs)
    clustering(params.config, eigdecomp.out.eigvals, eigdecomp.out.eigvecs, make_bintable.out.bins)
    heatmap(params.config, eigdecomp.out.eigvals, eigdecomp.out.eigvecs, make_bintable.out.bins, clustering.out.cluster, make_track_db.out.trackdb, params.meta)
    scatters(params.config, eigdecomp.out.eigvals, eigdecomp.out.eigvecs, make_bintable.out.bins, clustering.out.cluster, make_track_db.out.trackdb, params.meta)
}
