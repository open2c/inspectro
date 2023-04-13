#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process plot_spectrum {

    label 'process_single'
    container = 'tariship/inspectrov1@sha256:90ebe2ee28a5b6bf7aa67e2abfdfd82f93dd39e02ce37f9bdbdf1e12ba1db3e6'
    publishDir "${params.outdir}/figures"
    
    input:
        path config
        path eigvals

    output:
        path "*.pdf"

    script:

    """ 
    plot_spectrum.py --config ${config} --eigvals ${eigvals}
    """ 
}