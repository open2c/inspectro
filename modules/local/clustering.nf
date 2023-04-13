#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process clustering {
    
    label 'process_medium'
    container = 'tariship/inspectrov1@sha256:90ebe2ee28a5b6bf7aa67e2abfdfd82f93dd39e02ce37f9bdbdf1e12ba1db3e6'
    publishDir "${params.outdir}"

    input:
        path config
        path eigvals
        path eigvecs
        path bins

    output:
        path "*.tsv", emit: cluster

    script:

    """ 
    clustering_main.py --config ${config} --eigvals ${eigvals} --eigvecs ${eigvecs} --bins ${bins}
    """
}