import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plotMap(self, module, maptype):
    # Plot all spaxels, or only those in Voronoi-region
    if self.restrict2voronoi == 2:
        idxMap = np.where( self.table.BIN_ID >= 0 )[0]
    else:
        idxMap = np.arange( len(self.table.BIN_ID) )

    self.currentMapQuantity = maptype

    # Voronoi table
    if module == 'TABLE':
        val = self.table[maptype][idxMap]
        if maptype == 'FLUX': val = np.log10(val)

    # Masking
    if module == 'MASK':
        val = self.Mask[maptype][idxMap]

    # stellarKinematics
    if module == 'KIN':
        val = self.kinResults[maptype][idxMap]

    # emissionLines
    if module == 'GAS':
        val = self.gasResults[maptype][idxMap]

        try:
            idx_AoNThreshold = np.where( self.gasResults[maptype.split('_')[0]+'_'+maptype.split('_')[1]+'_AON'][idxMap] < self.AoNThreshold )[0]
            val[idx_AoNThreshold] = np.nan
        except:
            logging.info("WARNING: No AoN threshold is applied to the displayed map of "+maptype)

    # starFormationHistories
    if module == 'SFH':
        if maptype.split('_')[-1] == 'DIFF':
            val = self.kinResults[maptype.split('_')[0]][idxMap] - self.sfhResults[maptype.split('_')[0]][idxMap]
        else:
            val = self.sfhResults[maptype][idxMap]

    # lineStrengths
    if module == 'LS':
        val = self.lsResults[maptype][idxMap]



    # Create image in pixels
    xmin = np.nanmin(self.table.X[idxMap])-1;  xmax = np.nanmax(self.table.X[idxMap])+1
    ymin = np.nanmin(self.table.Y[idxMap])-1;  ymax = np.nanmax(self.table.Y[idxMap])+1
    npixels_x = int( np.round( (xmax - xmin)/self.pixelsize ) + 1 )
    npixels_y = int( np.round( (ymax - ymin)/self.pixelsize ) + 1 )
    i = np.array( np.round( (self.table.X[idxMap] - xmin)/self.pixelsize ), dtype=int )
    j = np.array( np.round( (self.table.Y[idxMap] - ymin)/self.pixelsize ), dtype=int )
    image = np.full( (npixels_x, npixels_y), np.nan )
    image[i,j] = val

    # Define special labels
    if   maptype == 'FLUX':    clabel="log( Flux )"  ; cmap='plasma'
    elif maptype == 'V':       clabel="v [km/s]"     ; cmap='RdBu'
    elif maptype == 'SIGMA':   clabel="sigma [km/s]" ; cmap='plasma'
    elif maptype == 'AGE':     clabel="Age [Gyr]"    ; cmap='plasma'
    elif maptype == 'METALS':  clabel="[M/H]"        ; cmap='plasma'
    elif maptype == 'ALPHA':   clabel="[Mg/Fe]"      ; cmap='plasma'
    elif maptype == 'BIN_ID':  clabel="BIN_ID"      ; cmap='plasma'
    else:                      clabel=maptype        ; cmap='plasma'

    # x = np.arange(xmin-self.pixelsize/2, xmax+self.pixelsize/2, self.pixelsize)[:npixels_x]
    # y = np.arange(ymin-self.pixelsize/2, ymax+self.pixelsize/2, self.pixelsize)[:npixels_y]
    x = np.arange(xmin, xmax + self.pixelsize, self.pixelsize)[:npixels_x]
    y = np.arange(ymin, ymax + self.pixelsize, self.pixelsize)[:npixels_y]
    fig = px.imshow(np.rot90(image)[::-1],
                    x=x,
                    y=y,
                    labels={'x': 'x [arcsec]', 'y':'y [arcsec]', 'color': clabel},
                    color_continuous_scale=cmap,
                    aspect='equal')
    # fig = go.Figure(data=go.Heatmap(
    #                            z=np.rot90(image)[::-1],
    #                            x=x,
    #                            y=y,
    #                             labels={'x': 'x [arcsec]', 'y':'y [arcsec]', 'color': clabel}))

    if hasattr(self, 'idxBinShort') == True and self.idxBinShort >= 0:
        fig.add_trace(go.Scatter(x=[self.table.XBIN[self.table['BIN_ID']==self.idxBinShort][0]], y=[self.table.YBIN[self.table['BIN_ID']==self.idxBinShort][0]], opacity=0.6,
                                 mode='markers', name='VorBin', marker=dict(symbol='x', line_width=1.5, line_color='white', color='black', size=8)))
    else:
       fig.add_trace(go.Scatter(x=None, y=None, opacity=0.6,
                         mode='markers', name='VorBin', marker=dict(symbol='x', line_width=1.5, line_color='white', color='black', size=8)))
    if hasattr(self, 'idxBinLong') == True:
        # if self.idxBinLong != None:
            fig.add_trace(go.Scatter(x=[self.table.X[self.idxBinLong]], y=[self.table.Y[self.idxBinLong]], opacity=0.6,
                                     mode='markers', name='SpaxelBin', marker=dict(symbol='circle', line_width=1.5, line_color='white', color='black', size=8)))
    else:
        fig.add_trace(go.Scatter(x=None, y=None, opacity=0.6,
                         mode='markers', name='SpaxelBin', marker=dict(symbol='circle', line_width=1.5, line_color='white', color='black', size=8)))


    # print(np.nanpercentile(image, [10, 95]))
    if module not in ['TABLE', 'MASK']:
        fig.update_coloraxes(cmin=np.nanpercentile(image, 5), cmax=np.nanpercentile(image, 95))
    if module == 'MASK':
        fig.update_coloraxes(cmin=0, cmax=1)
    if maptype == 'V':
        absmax = np.nanpercentile(np.abs(image), 95)
        fig.update_coloraxes(cmin=-absmax, cmax=absmax, cmid=0)
    fig.update_yaxes(autorange=True)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False, hoverdistance=2)
    # fig.update_layout(coloraxis_colorbar_x=-0.1)

    self.fig_plotMap = fig

    return self.fig_plotMap


# ============================================================================ #
#                              P L O T   D A T A                               #
# ============================================================================ #
def plotSpectra(self):
    figs = []

    # Plot stellarKinematics fit
    if self.KIN == True:
        fig = plotSpectraKIN(self, self.Spectra[self.idxBinShort], self.kinBestfit[self.idxBinShort], self.kinGoodpix)
        if 'V' in self.kinResults.names  and  'SIGMA' in self.kinResults.names  and  'H3' in self.kinResults.names  and  'H4' in self.kinResults.names:
            fig.update_layout(title={'text': "Stellar kinematics: V={:.2f}km/s, SIGMA={:.2f}km/s, H3={:.3f}, H4={:.3f}".format(self.kinResults_Vorbin.V[self.idxBinShort], self.kinResults_Vorbin.SIGMA[self.idxBinShort], self.kinResults_Vorbin.H3[self.idxBinShort], self.kinResults_Vorbin.H4[self.idxBinShort]),
                                     'x': 0.50, 'y':0.92,
                                     'xanchor':'center', 'yanchor':'top'})
        elif 'V' in self.kinResults.names  and  'SIGMA' in self.kinResults.names:
            fig.update_layout(title={'text': "Stellar kinematics: V={:.2f}km/s, SIGMA={:.2f}km/s".format(self.kinResults_Vorbin.V[self.idxBinShort], self.kinResults_Vorbin.SIGMA[self.idxBinShort]),
                                     'x': 0.50, 'y':0.92,
                                     'xanchor':'center', 'yanchor':'top'})
        figs.append(fig)

    # Plot emissionLines fit
    if self.GAS == True:
        if self.gasLevel == 'BIN':
            fig = plotSpectraGAS(self, self.Spectra[self.idxBinShort], self.gasBestfit[self.idxBinShort], self.gasGoodpix)
        elif self.gasLevel == 'SPAXEL':
            fig = plotSpectraGAS(self, self.AllSpectra[self.idxBinLong], self.gasBestfit[self.idxBinLong], self.gasGoodpix)
            # self.axes[2].set_title("SPAXEL_ID = {:d}".format(self.self.idxBinLong), loc='right')
        try:
            line='0' # need to modify for the future
            fig.update_layout(title={'text': "Emission-line kinematics: v={:.2f}km/s, sigma={:.2f}km/s".format(self.gasResults_Vorbin[line+'_V'][self.idxBinShort], self.gasResults_Vorbin[line+'_S'][self.idxBinShort]),
                                     'x': 0.50, 'y':0.92,
                                     'xanchor':'center', 'yanchor':'top'})
        except:
            fig.update_layout(title={'text': 'Emission-line analysis',
                                     'x': 0.50, 'y':0.92,
                                     'xanchor':'center', 'yanchor':'top'})
        figs.append(fig)

    # Plot starFormationHistories results
    if self.SFH == True:
        fig = plotSpectraSFH(self, self.Spectra[self.idxBinShort], self.sfhBestfit[self.idxBinShort], self.sfhGoodpix)
        fig.update_layout(title={'text': "Stellar populations: Age={:.2f} Gyr, [M/H]={:.2f}, [alpha/Fe]={:.2f}".format(self.sfhResults_Vorbin['AGE'][self.idxBinShort], self.sfhResults_Vorbin['METAL'][self.idxBinShort], self.sfhResults_Vorbin['ALPHA'][self.idxBinShort]),
                                 'x': 0.50, 'y':0.92,
                                 'xanchor':'center', 'yanchor':'top'})
        figs.append(fig)

    return figs


def plotSpectraKIN(self, spectra, bestfit, goodpix):

    # Compile information on masked regions
    masked = np.flatnonzero( np.abs(np.diff(goodpix)) > 1)
    vlines = []
    for i in masked:
        vlines.append( goodpix[i]+1 )
        vlines.append( goodpix[i+1]-1 )
    vlines = np.array(vlines)


    # Offset of residuals
    offset = np.min(bestfit[:]) - (np.max(bestfit[:]) - np.min(bestfit[:]))*0.10

    # Plot spectra
    idxMin = np.where( self.Lambda == self.kinLambda[0]  )[0][0]
    idxMax = np.where( self.Lambda == self.kinLambda[-1] )[0][0]
    idxLam = np.arange(idxMin, idxMax+1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=self.LambdaLIN[idxLam], y=spectra[idxLam],
                             name='Spectrum', mode='lines', line=dict(color='black',     width=2)))
    fig.add_trace(go.Scatter(x=self.kinLambdaLIN,      y=bestfit[:],
                             name='Bestfit',  mode='lines', line=dict(color='crimson',   width=2)))
    fig.add_trace(go.Scatter(x=self.kinLambdaLIN,      y=spectra[idxLam] - bestfit + offset,
                             name='Residual', mode='lines', line=dict(color='limegreen', width=2)))

    # Highlight masked regions
    i = 0
    while i < len(vlines)-1:
        badpix = np.arange(vlines[i],vlines[i+1]+1)
        i += 2
    fig.add_trace(go.Scatter(x=[self.LambdaLIN[idxLam][0], self.LambdaLIN[idxLam][-1]], y=[offset,offset],
                             name=None, mode='lines', line=dict(color='black',     width=1)))

    shapes = []
    for i in range( len(np.where(vlines != 0)[0]) ):
        if i%2 == 0:
            x0 = self.LambdaLIN[idxLam][vlines[i]]
            x1 = self.LambdaLIN[idxLam][vlines[i+1]]
            shapes.append({
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': x0,
                'x1': x1,
                'y0': 0,
                'y1': 1,
                'fillcolor': 'grey',
                'opacity': 0.1,
                'line': {'width': 0}
            })
    fig.update_layout(shapes=shapes)

    fig.update_layout(xaxis=dict(range=[self.LambdaLIN[idxLam][0], self.LambdaLIN[idxLam][-1]]),
                      xaxis_title='Wavelength (Angstrom)', yaxis_title='Flux', showlegend=False,
                      margin=dict(l=0, r=0, t=35, b=0))
    fig.update_layout(hovermode="x unified")

    return fig


def plotSpectraGAS(self, spectra, bestfit, goodpix):

    # Compile information on masked regions
    masked = np.flatnonzero( np.abs(np.diff(goodpix)) > 1)
    vlines = []
    for i in masked:
        vlines.append( goodpix[i]+1 )
        vlines.append( goodpix[i+1]-1 )
    vlines = np.array(vlines)


    # Offset of residuals
    offset = np.min(bestfit[:]) - (np.max(bestfit[:]) - np.min(bestfit[:]))*0.10

    # Plot spectra
    idxMin = np.where( self.Lambda == self.gasLambda[0]  )[0]
    idxMax = np.where( self.Lambda == self.gasLambda[-1] )[0]
    idxLam = np.arange(idxMin, idxMax+1)

    fig = go.Figure()
    if self.gasLevel == 'BIN':
        fig.add_trace(go.Scatter(x=self.gasLambdaLIN, y=self.EmissionSubtractedSpectraBIN[self.idxBinShort,:],
                             name='gasLevelBIN', mode='lines', line=dict(color='orange',     width=2)))
    elif self.gasLevel == 'SPAXEL':
        fig.add_trace(go.Scatter(x=self.gasLambdaLIN, y=self.EmissionSubtractedSpectraSPAXEL[self.idxBinLong,:],
                             name='gasLevelSPAXEL', mode='lines', line=dict(color='orange',     width=2)))

    fig.add_trace(go.Scatter(x=self.LambdaLIN[idxLam], y=spectra[idxLam],
                 name='Spectrum', mode='lines', line=dict(color='black',     width=2)))
    fig.add_trace(go.Scatter(x=self.gasLambdaLIN, y=bestfit[:],
                 name='Bestfit', mode='lines', line=dict(color='crimson',     width=2)))
    fig.add_trace(go.Scatter(x=self.gasLambdaLIN, y=spectra[idxLam] - bestfit + offset,
                 name='Residual', mode='lines', line=dict(color='limegreen',     width=2)))

    # Highlight masked regions
    i = 0
    while i < len(vlines)-1:
        badpix = np.arange(vlines[i],vlines[i+1]+1)
        i += 2
    fig.add_trace(go.Scatter(x=[self.LambdaLIN[idxLam][0], self.LambdaLIN[idxLam][-1]], y=[offset,offset],
                         name=None, mode='lines', line=dict(color='black',     width=1)))

    shapes = []
    for i in range( len(np.where(vlines != 0)[0]) ):
        if i%2 == 0:
            x0 = self.LambdaLIN[idxLam][vlines[i]]
            x1 = self.LambdaLIN[idxLam][vlines[i+1]]
            shapes.append({
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': x0,
                'x1': x1,
                'y0': 0,
                'y1': 1,
                'fillcolor': 'grey',
                'opacity': 0.1,
                'line': {'width': 0}
            })
    fig.update_layout(shapes=shapes)

    fig.update_layout(xaxis=dict(range=[self.LambdaLIN[idxLam][0], self.LambdaLIN[idxLam][-1]]),
                      xaxis_title='Wavelength (Angstrom)', yaxis_title='Flux', showlegend=False,
                      margin=dict(l=0, r=0, t=35, b=0))
    fig.update_layout(hovermode="x unified")

    return fig


def plotSpectraSFH(self, spectra, bestfit, goodpix):

    # Compile information on masked regions
    masked = np.flatnonzero( np.abs(np.diff(goodpix)) > 1)
    vlines = []
    for i in masked:
        vlines.append( goodpix[i]+1 )
        vlines.append( goodpix[i+1]-1 )
    vlines = np.array(vlines)

    # Offset of residuals
    offset = np.min(bestfit[:]) - (np.max(bestfit[:]) - np.min(bestfit[:]))*0.10

    # Plot spectra
    idxMin = np.where( self.Lambda == self.sfhLambda[0]  )[0]
    idxMax = np.where( self.Lambda == self.sfhLambda[-1] )[0]
    idxLam = np.arange(idxMin, idxMax+1)

    fig = go.Figure()
    try:
        idxMin     = np.where( self.gasLambda == self.sfhLambda[0]  )[0]
        idxMax     = np.where( self.gasLambda == self.sfhLambda[-1] )[0]
        idxLamGand = np.arange(idxMin, idxMax+1)
        fig.add_trace(go.Scatter(x=self.gasLambdaLIN[idxLamGand], y=self.EmissionSubtractedSpectraBIN[self.idxBinShort,idxLamGand],
                     name='EmissionSub', mode='lines', line=dict(color='orange',     width=2)))
        fig.add_trace(go.Scatter(x=self.gasLambdaLIN[idxLamGand], y=self.EmissionSubtractedSpectraBIN[self.idxBinShort,idxLamGand] - bestfit + offset,
                     name='Residual', mode='lines', line=dict(color='limegreen',     width=2)))
    except:
        pass

    fig.add_trace(go.Scatter(x=self.LambdaLIN[idxLam], y=spectra[idxLam],
                 name='Spectrum', mode='lines', line=dict(color='black',     width=2)))
    fig.add_trace(go.Scatter(x=self.sfhLambdaLIN, y=bestfit[:],
                 name='Bestfit', mode='lines', line=dict(color='crimson',     width=2)))

    # Highlight masked regions
    i = 0
    while i < len(vlines)-1:
        badpix = np.arange(vlines[i],vlines[i+1]+1)
        i += 2
    fig.add_trace(go.Scatter(x=[self.LambdaLIN[idxLam][0], self.LambdaLIN[idxLam][-1]], y=[offset,offset],
                         name=None, mode='lines', line=dict(color='black',     width=1)))

    shapes = []
    for i in range( len(np.where(vlines != 0)[0]) ):
        if i%2 == 0:
            x0 = self.LambdaLIN[idxLam][vlines[i]]
            x1 = self.LambdaLIN[idxLam][vlines[i+1]]
            shapes.append({
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': x0,
                'x1': x1,
                'y0': 0,
                'y1': 1,
                'fillcolor': 'grey',
                'opacity': 0.1,
                'line': {'width': 0}
            })
    fig.update_layout(shapes=shapes)

    fig.update_layout(xaxis=dict(range=[self.LambdaLIN[idxLam][0], self.LambdaLIN[idxLam][-1]]),
                      xaxis_title='Wavelength (Angstrom)', yaxis_title='Flux', showlegend=False,
                      margin=dict(l=0, r=0, t=35, b=0))
    fig.update_layout(hovermode="x unified")
    return fig


def make_html_subplots(database, module, maptype, persuffix, value):
    fig_plotMap = plotMap(database, module, maptype)
    figs_spec = plotSpectra(database)

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{'rowspan': 3, 'colspan': 1}, {'colspan': 1}],
            [None, {'colspan': 1}],
            [None, {'colspan': 1}],
        ],
    vertical_spacing=0.12,
    horizontal_spacing=0.12,
    subplot_titles=[" ", " ", " ", " "])

    trace1 = go.Scatter(
               x=np.linspace(0, 2, 60),
               y=np.random.rand(60),
               mode='lines',
               line=dict(width=1, color='red')
               )


    for trace_i in fig_plotMap.data:
        fig.add_trace(trace_i, row=1, col=1)
    fig['layout']['xaxis'].update(title=fig_plotMap.layout['xaxis']['title']['text'])
    fig['layout']['yaxis'].update(title=fig_plotMap.layout['yaxis']['title']['text'])
    fig['layout']['yaxis']['scaleanchor']='x'
    fig.layout.annotations[0].update(text='%s Map (%s, BIN_ID=%s, value=%.3f)' % (maptype, persuffix, str(database.idxBinShort), value))

    fig_spec_row = 1
    for fig_spec in figs_spec:
        for trace_i in fig_spec.data:
            fig.add_trace(trace_i, row=fig_spec_row, col=2)
        fig['layout']['xaxis%s' % str(fig_spec_row+1)].update(fig_spec.layout['xaxis'])
        fig['layout']['yaxis%s' % str(fig_spec_row+1)].update(fig_spec.layout['yaxis'])
        fig.layout.annotations[fig_spec_row].update(text=fig_spec.layout['title']['text'])
        fig_spec_row += 1

    fig.update_layout(coloraxis = fig_plotMap.layout['coloraxis'].update({'colorbar_x': 0.45}))
    fig.update_layout(showlegend=False, margin=dict(l=50, r=50, t=50, b=50))
    return fig
