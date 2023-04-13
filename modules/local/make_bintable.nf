#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process make_bintable {

    label 'process_single'
    container = 'tariship/inspectrov1@sha256:90ebe2ee28a5b6bf7aa67e2abfdfd82f93dd39e02ce37f9bdbdf1e12ba1db3e6'
    publishDir "${params.outdir}"
        
    input:        
        path config
        path fasta

    output:
        path "*.bed", emit: chromarms
        path "*.pq", emit: bins

    script:

    """ 
    make_bintable.py --config ${config} --genome ${fasta}
    """ 
}