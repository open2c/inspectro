#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process eigdecomp {
    
    label 'process_high'
    container = 'tariship/inspectrov1@sha256:90ebe2ee28a5b6bf7aa67e2abfdfd82f93dd39e02ce37f9bdbdf1e12ba1db3e6'
    publishDir "${params.outdir}"

    input:
        path config
        path bins
        path mcool

    output:
        path "*.eigvals.pq", emit: eigvals
        path "*.eigvecs.pq", emit: eigvecs

    script:

    """ 
    eigdecomp_main.py --config ${config} --bins ${bins} --cooler ${mcool}
    """ 
}