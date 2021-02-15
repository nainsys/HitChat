module.exports = function(app, fs)
{
     app.get('/',function(req,res){
         res.render('index', {
             title: "한국IT 챗봇"
         })
     });
}